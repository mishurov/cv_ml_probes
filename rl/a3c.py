#!/usr/bin/env python

import multiprocessing
import threading

from absl import flags
import tensorflow as tf
import numpy as np
import gym


flags.DEFINE_string('env', 'BipedalWalker-v2', 'Gym environment name.')
flags.DEFINE_float('beta', 0.005, 'Entropy beta.')
flags.DEFINE_float('gamma', 0.965, 'Discount rate.')
flags.DEFINE_integer('max_episodes', 40000, 'Maximum number of episodes.')
flags.DEFINE_float('actor_lr', 0.00005, 'Actor learning rate.')
flags.DEFINE_float('critic_lr', 0.0001, 'Critic learning rate.')
flags.DEFINE_integer('update_iter', 10,
                     'Number of itersations between updates.')
flags.DEFINE_string('train_log_dir', '/tmp/a3c',
                    'Directory where to write event logs.')
flags.DEFINE_integer('seed', 25, 'Global random seed.')
flags.DEFINE_string(
    'eval_model', '',
    'Path to a particular model. '
    'If set, the app will evaluate network instead of training.'
    'If the model was not found, Random Agent is used for evaluation.')


FLAGS = flags.FLAGS


class ActorCriticNetwork(object):
  def __init__(self, scope, opt_actor, opt_critic, global_net=None):
    env = gym.make(FLAGS.env)
    self.action_dim = env.action_space.shape[0]
    action_bound = [env.action_space.low, env.action_space.high]
    state_dim = env.observation_space.shape[0]

    with tf.variable_scope(scope):
      self.state = tf.placeholder(tf.float32, [None, state_dim], 'state')
      if global_net is None:
        self.create_network()
        self.actor_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor'
        )
        self.critic_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic'
        )
      else:
        self.action_in = tf.placeholder(
            tf.float32, [None, self.action_dim], 'action'
        )
        self.value_target = tf.placeholder(tf.float32, [None, 1], 'value_target')
        mu, sigma, self.value = self.create_network()

        self.td_error = tf.subtract(self.value_target, self.value, name='td_error')
        with tf.name_scope('critic_loss'):
          self.critic_loss = tf.reduce_mean(tf.square(self.td_error))
        with tf.name_scope('wrap_action_out'):
          self.test = sigma[0]
          mu, sigma = mu * action_bound[1], sigma + 1e-5

        normal_dist = tf.contrib.distributions.Normal(mu, sigma)
        with tf.name_scope('actor_loss'):
          log_prob = normal_dist.log_prob(self.action_in)
          exp_v = log_prob * self.td_error
          entropy = normal_dist.entropy()
          self.exp_v = FLAGS.beta * entropy + exp_v
          self.actor_loss = tf.reduce_mean(-self.exp_v)

        with tf.name_scope('choose_action'):
          self.action = tf.clip_by_value(
              tf.squeeze(normal_dist.sample(1)), *action_bound
          )
        with tf.name_scope('local_gradients'):
          self.actor_params = tf.get_collection(
              tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor'
          )
          self.critic_params = tf.get_collection(
              tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic'
          )
          self.actor_gradients = tf.gradients(self.actor_loss, self.actor_params)
          self.critic_gradients = tf.gradients(self.critic_loss, self.critic_params)

    if global_net is not None:
      with tf.name_scope('sync'):
        with tf.name_scope('pull'):
          self.pull_actor = [
              lp.assign(gp) for lp, gp in zip(self.actor_params,
                                              global_net.actor_params)
          ]
          self.pull_critic = [
              lp.assign(gp) for lp, gp in zip(self.critic_params,
                                              global_net.critic_params)
          ]
        with tf.name_scope('push'):
          self.update_actor = opt_actor.apply_gradients(
              zip(self.actor_gradients, global_net.actor_params)
          )
          self.update_critic = opt_critic.apply_gradients(
              zip(self.critic_gradients, global_net.critic_params)
          )

  def create_network(self):
    w_init = tf.contrib.layers.xavier_initializer()
    with tf.variable_scope('actor'):
      la = tf.layers.dense(self.state, 500, tf.nn.relu6,
                           kernel_initializer=w_init, name='layer_1')
      la = tf.layers.dense(la, 300, tf.nn.relu6,
                           kernel_initializer=w_init, name='layer_2')
      la = tf.layers.dense(la, 100, tf.nn.relu6,
                           kernel_initializer=w_init, name='layer_3')
      mu = tf.layers.dense(la, self.action_dim, tf.nn.tanh,
                           kernel_initializer=w_init, name='mu')
      sigma = tf.layers.dense(la, self.action_dim, tf.nn.softplus,
                              kernel_initializer=w_init, name='sigma')
    with tf.variable_scope('critic'):
      lc = tf.layers.dense(self.state, 500, tf.nn.relu6,
                           kernel_initializer=w_init, name='layer_1')
      lc = tf.layers.dense(lc, 300, tf.nn.relu6,
                           kernel_initializer=w_init, name='layer_2')
      lc = tf.layers.dense(lc, 100, tf.nn.relu6,
                           kernel_initializer=w_init, name='layer_3')
      value = tf.layers.dense(lc, 1, kernel_initializer=w_init, name='value')
    return mu, sigma, value

  def update_global(self, sess, feed_dict):
    _, _, t = sess.run(
        [self.update_actor, self.update_critic, self.test], feed_dict
    )
    return t

  def pull_global(self, sess):
    sess.run([self.pull_actor, self.pull_critic])

  def choose_action(self, sess, s):
    s = s[np.newaxis, :]
    return sess.run(self.action, {self.state: s})


class Worker(object):
  def __init__(self, name, opt_actor, opt_critic, global_episodes,
               global_running_reward, global_network, seed):
    self.env = gym.make(FLAGS.env)
    self.env.seed(seed)
    self.name = name
    self.net = ActorCriticNetwork(name, opt_actor, opt_critic, global_network)
    self.summary_writer = tf.summary.FileWriter(
        FLAGS.train_log_dir + "/" + self.name
    )
    self.global_episodes = global_episodes
    self.increment = self.global_episodes.assign_add(1, use_locking=True)

    self.global_running_reward = global_running_reward
    self.latest_reward = tf.placeholder(tf.float32, [], 'latest_reward')
    self.assign_reward = self.global_running_reward.assign(
        self.latest_reward, use_locking=True
    )

  def work(self, sess, coord, saver):
    total_step = 1
    buffer_state, buffer_action, buffer_reward = [], [], []
    episode_count = sess.run(self.global_episodes)
    episode_rewards, episode_positions = [], []

    while not coord.should_stop() and episode_count < FLAGS.max_episodes:
      state = self.env.reset()
      episode_reward = 0
      while True:
        action = self.net.choose_action(sess, state)
        next_state, reward, done, _ = self.env.step(action)
        if reward == -100:
          reward = -2

        episode_reward += reward
        buffer_state.append(state)
        buffer_action.append(action)
        buffer_reward.append(reward)

        if total_step % FLAGS.update_iter == 0 or done:
          if done:
            value = 0
          else:
            value = sess.run(
                self.net.value, {self.net.state: next_state[np.newaxis, :]}
            )[0, 0]

          buffer_value_target = []
          for r in buffer_reward[::-1]:
            value = r + FLAGS.gamma * value
            buffer_value_target.append(value)

          buffer_value_target.reverse()

          buffer_state = np.vstack(buffer_state)
          buffer_action = np.vstack(buffer_action)
          buffer_value_target = np.vstack(buffer_value_target)

          feed_dict = {
              self.net.state: buffer_state,
              self.net.action_in: buffer_action,
              self.net.value_target: buffer_value_target,
          }

          self.net.update_global(sess, feed_dict)
          buffer_state, buffer_action, buffer_reward = [], [], []
          self.net.pull_global(sess)

        state = next_state
        total_step += 1

        if (done and episode_count % 100 == 0 and len(episode_positions) and
            len(episode_rewards)):
          summary = tf.Summary()
          episode_positions = episode_positions[-100:]
          episode_rewards = episode_rewards[-100:]
          mean_reward = np.mean(episode_rewards)
          summary.value.add(tag="Episode Reward",
                            simple_value=float(mean_reward))
          mean_position = np.mean(episode_positions)
          summary.value.add(tag="Position", simple_value=float(mean_position))
          self.summary_writer.add_summary(summary, episode_count)

        if done:
          global_running_reward = sess.run(self.global_running_reward)
          episode_count = sess.run(self.global_episodes)
          sess.run(self.increment)
          if episode_count == 0:
            latest_reward = episode_reward
          else:
            latest_reward = 0.95 * global_running_reward + 0.05 * episode_reward
          sess.run(self.assign_reward, {self.latest_reward: latest_reward})
          if (episode_count != 0 and total_step > 2 and
              ((episode_count % 1000 == 0 and episode_count <= 10000) or
               (episode_count % 2500 == 0 and episode_count > 10000))):
            tf.logging.info("Saving checkpoints for {} into {}".format(
                episode_count, FLAGS.train_log_dir))
            saver.save(
                sess,
                FLAGS.train_log_dir + '/model.ckpt-' + str(episode_count)
            )
          pos = self.env.unwrapped.hull.position[0]
          # print calls in threads seeem to work faster than tf.logging
          print("{}, ep {}, pos {:.1f}, r {:.1f}, er {:.1f}".format(
                self.name, episode_count, pos, latest_reward, episode_reward))
          episode_rewards.append(episode_reward)
          episode_positions.append(pos)
          break


def init_or_restore_variables(sess, saver):
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_log_dir)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)
  else:
    sess.run(tf.global_variables_initializer())


def create_global_network():
  opt_actor = tf.train.RMSPropOptimizer(
      FLAGS.actor_lr, name="optimizer_actor", decay=0.95, use_locking=True
  )
  opt_critic = tf.train.RMSPropOptimizer(
      FLAGS.critic_lr, name="optimizer_critic", decay=0.95, use_locking=True
  )
  global_network = ActorCriticNetwork("global_network", opt_actor, opt_critic)
  return opt_actor, opt_critic, global_network


def train(sess):
  global_episodes = tf.get_variable(
      "global_episodes", [], dtype=tf.int32, trainable=False
  )
  global_running_reward = tf.get_variable(
      "global_running_reward", [], dtype=tf.float32, trainable=False
  )
  with tf.device("/cpu:0"):
    opt_actor, opt_critic, global_network = create_global_network()
    workers = []
    num_workers = multiprocessing.cpu_count()
    global_seed = tf.get_default_graph().seed
    for i in range(num_workers):
      name = "worker_{}".format(i)
      workers.append(
          Worker(name, opt_actor, opt_critic, global_episodes,
                 global_running_reward, global_network, global_seed + i)
      )

  coord = tf.train.Coordinator()
  saver = tf.train.Saver(max_to_keep=150)
  init_or_restore_variables(sess, saver)

  tf.summary.FileWriter(FLAGS.train_log_dir + "/graph", sess.graph)

  worker_threads = []

  for worker in workers:
    t = threading.Thread(target=worker.work, args=(sess, coord, saver))
    t.start()
    worker_threads.append(t)

  coord.join(worker_threads)


def evaluate(sess):
  from gym.envs.classic_control import rendering

  env = gym.make(FLAGS.env)
  env.env.viewer = rendering.Viewer(640, 480)
  env.env.viewer.window.set_visible(False)
  viewer = rendering.SimpleImageViewer()

  blank = np.zeros((10, 10, 3), dtype=np.uint8)
  viewer.imshow(blank)
  viewer.window.set_caption("A3C")
  viewer.window.set_size(960, 720)
  viewer.window.set_location(480, 200)

  @viewer.window.event
  def on_key_press(symbol, modifiers):
    if symbol in [113, 65307]:
      exit()
    else:
      env.reset()

  @viewer.window.event
  def on_close():
    exit()

  exists = tf.train.checkpoint_exists(FLAGS.eval_model)
  if exists:
    opt_actor, opt_critic, global_network = create_global_network()
    net = ActorCriticNetwork("worker_0", opt_actor, opt_critic, global_network)
    saver = tf.train.Saver()
    saver.restore(sess, FLAGS.eval_model)
  else:
    tf.logging.info("Using Random Agent")

  while True:
    state = env.reset()
    viewer.done = False
    while not viewer.done:
      rgb = env.render(mode='rgb_array')
      viewer.imshow(rgb)
      if exists:
        action = net.choose_action(sess, state)
      else:
        action = env.action_space.sample()
      state, _, viewer.done, _ = env.step(action)


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  sess = tf.Session()
  tf.set_random_seed(FLAGS.seed)
  evaluate(sess) if FLAGS.eval_model else train(sess)


if __name__ == '__main__':
  tf.app.run()
