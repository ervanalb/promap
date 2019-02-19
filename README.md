# promap
Python projection mapping pipeline

## What is it?

`promap` is a command-line tool that you can use
in conjunction with your digital video workflow
to do projection mapping.

`promap` uses a projector and a single camera
to compute the physical scene as viewed from the projector
as well as a coarse disparity (depth) map.

You can import the maps that it generates straight into your digital video workflow,
or you can post-process them into masks and uvmaps first using image processing tools.

## How does it work?

1. Generate gray code images
2. Project the gray code images
3. Capture the images using the camera
4. Decode the images into a lookup table that goes from camera to projector space
5. Invert the lookup table so it goes from projector to camera space
6. Apply the inverted lookup table to the camera's view of the scene, producing the projector's view of the scene
7. Compute a disparity map of the lookup table using least-squares fitting

`promap` will automatically perform all of these steps for you.
The scanning only takes about a minute.

You can also perform a subset of these tasks,
either to introspect the data,
or to tweak the pipeline to your satisfaction.
The output of any of steps can be saved,
and the input of any step can be loaded.

## What is the status of the project?

A proof of concept works (in the `old` folder) but it is in the process of being overhauled. It is not ready for use yet.
