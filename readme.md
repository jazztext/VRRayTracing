# GPU Accelerated RayTracing for VR
Chris Kaffine and Zach Shearer

##Summary
We will be writing a fully functional raytracer in CUDA to render photorealistic images to an Oculus Rift in real time.

##Background

##The Challenge

##Resources
We will use the raytracing code we wrote for graphics last semester as a starting point, but as we port it to the GPU we
expect most of it will need to be heavily rewritten. We already have an Oculus to work with, and for most of the project we
will run our code on our own laptops. We will be using  as references to guide the design of our GPU implementation. At the end
of the project we would like to run our code on a more powerful GPU than we have in our laptops, but we would need physical
access to the machine to use the Oculus, and it needs to run Windows. If we can't get access to such a machine we will run
our code on Latedays without rendering to the Oculus so we can still get performance measurements.

##Goals

The primary goal that we plan to achieve is a system capable of rendering images of a reasonable quality to the Oculus in real
time. Specifically this means being able to run at 60fps, which means the whole rendering pipeline must have a latency under
16ms. We expect to achieve this performance on relatively simple scenes with potentially complex indirect lighting effects. If
we make quick progress on the project, we hope to improve the quality of our output by implementing a more advanced renderer,
though we will most likely keep the frame rate at 60fps. At the parallelism competition we will show a demo of a scene being
rendered in real time on the Oculus. We will also show results on how quickly our system runs on input with varying complexity.

##Platform
Most of our code will be written in CUDA and run on Nvidia GPUs.

## Schedule

**Week 1:** Begin porting existing CPU raytracing code over to the GPU. By the end of the week we should have ray-scene 
intersections and direct lighting for simple diffuse materials implemented naively on the GPU. We will also set up the
software tools for the Oculus, with the goal of being able to render images to it by the end of the week.

**Week 2:** Finish basic raytracer implementation, including indirect lighting and mirror and glass materials. By the
checkpoint at the end of this week we should have a fully functional raytracer able to render to the Oculus, albeit at an
unacceptably low frame rate.

**Week 3:** Analyze the performance of the raytracer to identify bottlenecks, determine if the program is bandwidth bound or
compute bound, etc, and begin working on optimizations.

**Week 4:** Finish optimizing performance for the GPU, look into optimizations we could make specific to VR, such as modelling
the lense distortion and chromatic aberration during ray tracing, or rendering with lower resolution at the periphery.

**Week 5:** Finish any remaining optimizations, and run code on a more powerful GPU to get more realistic measurements. This
may involve tuning the code for the new platform, and if we run on Latedays we will need to write some sort of testing harness
to simulate the conditions of actually running on the Oculus.

**Time Permitting:** Extend the raytracer to use a more advanced rendering algorithm, such as bidirectional pathtracing or
photon mapping.
