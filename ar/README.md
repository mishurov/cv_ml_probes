# Feature detection and matching using SIFT

![matrix](http://mishurov.co.uk/images/github/cv_ml_probes/ar.png)

The project is based on the code from this repository https://github.com/MasteringOpenCV/code/tree/master/Chapter3_MarkerlessAR

The differences are more compact code, programmable OpenGL pipeline, ability to use ROI for choosing initial image, SIFT instead of ORB and FREAK, setting FoV and aspect ratio for building a camera matrix (the result is less precise but it doesn't require calibration).

I recommend to generate a project in "build" directory sicnce the source file refer to the sample video using the relative path.


