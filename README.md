# GPU Accelerated RayTracing for VR
Chris Kaffine and Zach Shearer

#Final Writeup
We implemented a GPU accelerated raytracer in CUDA to be used in VR applications, utilizing some VR specific optimizations.

##Background
Raytracing is a way of simulating realistic light interacting with a scene, in order to create more realistic looking images than using rasterization methods. Given as inputs is a scene, consisting of objects and light sources, and a camera, outputting an image simulating what this scene would look like. At its core, there are 7 steps to raytracing:

1. Sample a ray from the camera
2. Cast the ray into the scene and find where it intersects
3. Sample a ray from the intersection point to the light
4. Cast this shadow ray into the scene to see if the light source is blocked
5. Compute the radiance along the ray path
6. Probabilistically determine if the ray path should continue being evaluated
7. If so, sample a reflection ray and go back to step 2.

The computationally expensive part of raytracing is that there must be at least as many of these camera rays as pixels on the screen, which for an Oculus DK2 is over 2 million camera rays, and this will generate very low quality images. However, raytracing lends itself well to parallelization, in that each of these cameras can be evaluated completely independently of each other. However, bacause of the way that ray-scene intersections are tested (using a Bounding Volume Hierarchy), some rays can short-circuit this intersection test, which can potentially cause a lot of divergent execution. Additionally, rays will hit all different parts of the scenes, and will hit even more when reflected, which causes the need for a high number of incoherent memory accesses. Finally, VR settings require very low latency, typically the desired frame rates fall in the 75-90 fps range.

##Approach
We started out with a CPU implementation of raytracing (from 15-462), and ported it to run on a GPU using CUDA. This was done by mapping each pixel to a CUDA thread, where each pixel consisted of evaluating a user-specified number of rays. 
Our first attempt at improving performance was to try to reduce divergent execution by pruning inactive ray paths. This was done by maintaining a global pool of active rays that threads draw from to process. However, this system requires some overhead in synchronization and sorting, which turned out to be too much, given the low latency requirements of VR. Additionally, after some profiling, we determined that the bottleneck in the sysytem is the incoherent memory accesses, not divergent execution.
Our next approach was to try to reduce incoherent memory accesses. To do this, we changed many of our data structures to use CUDA vector types, to take advantage of more efficient vector load/store instructions. Also, we switched our arrays of structures to be structures of arrays, to have all threads in a warp accessing a contiguous block of memory when they all access the same field in a structure. Finally, we cached our frequently updated data structures in shared memory to reduce global memory traffic.
Another very useful optimization we did was to limit the register count in our main kernel to 64, after noticing that it initially required over 100. With the initial register count, the number of blocks that could run on the device at the same time was low, so we were missing out on most of the latency hiding abilities of a GPU. By reducing this to 64, we were able to fit more blocks on the device at the same time, which hid the latency much better, giving over a 2x speedup.
Finally, one more optimization we did was to decrease image quality around the edges of the image, since the user is more likely to be looking at the center of the screen, they will rarely notice the difference. The center was chosen as an approximation of foveated rendering, since actual eye-tracking software is not yet available.

##Results
We weren't able to test our full rendering code on a powerful machine like latedays, since it requires physical access to the machine and to install many libraries, unfortunately. However, on our testing laptop (NVIDIA 750M), we were able to render small scenes at medium-low quality at around 30 fps. It's worth noting that on single-frame tests, the latedays machines regularly got a 3-4x performance increase over our test machine, so on hardware that would typically be used to run VR systems, we would almost certainly hit the latency requirement.
Unfortunately, on larger scenes, the number of incoherent memory accesses makes performance drop off quickly with the number of objects in the scene.

##References
CHRIS. DO THIS SECTION PLEASE.

Equal work was performed by both project members.



#Parallelism Competition Update
At this point in time, we've implemented a more intelligent GPU raytracer, with an improved BVH traversal method, where all threads in a warp traverse until they reach a leaf. At that time, they all check over all primitives in the leaf, which improved performance on scenes with larger BVHs, as it helps reduce divergent execution. We did extensive performance analysis to determine the main bottlenecks of the naive raytracing implementation, and found that the largest was the ray-triangle intersection code. This was because it generated a large number of non-contiguous memory requests, which while it didn't approach the maximum bandwidth, the sheer volume created some issues with latency. To fix this, we've altered some of our data structures to help, and be able to utilize vector load instructions. Additionally, we've moved some of these to shared memory, and making a change from arrays of structures to structures of arrays. We also determined that the number of registers required by our kernel was limiting the occupancy of the kernel on the device, which reduced our latency hiding ability. To combat this, we determined what the best number of registers to use in our kernel was, and forced CUDA to use that number instead.
With all of these optimizations, we've acheived around a 6x speedup over our naive GPU implementation.

Upon testing on the actual Oculus hardware, everything is perfectly functional. Although, performance is not where we'd like it to be, and we believe that to be because of the less powerful GPU in our personal computers, which is all we can currently run the Oculus on. However, we did find that rendering to the Oculus uses a small amount of overhead, and rendering speeds are very similar to when we run the tests in headless mode, which is a promising result. Additionally, our tests have shown that the raytracing speed on the latedays machines will be around 3-4x faster than our personal computers, so the low framerate on our machines will become acceptable on those machines in our final tests.
Additionally, for the competition, we'll show a video of a scene being rendered in real time on simulated hardware as a demo.


##Checkpoint Summary
So far what we've completed is a functional, albeit unoptimized and slow raytracer in CUDA, and a functioning Oculus/OpenGL environment that's capable of interacting with CUDA code. At this time, the two are still separate tasks, and because of this, the code isn't on this page yet. We ran into some issues with getting the Oculus to work at first, like that our testing laptop's video card doesn't support any version of the SDK after 0.7, and then getting a reasonable development environment working on Windows. In the end, for ease of development, we rolled back to using an older version of the SDK that supports linux, and moving development over to a linux environment.

In terms of what we'll be able to produce at the end, we should still be able to do everything outlined in the original writeup, although at this point the stretch goals are looking unlikely. For the parallelism contest, our plan is to capture some test inputs (head movements, etc.), run the raytracer on a testing harness on latedays, and capture the output as a video. Since the latedays machines have the strongest GPUs of any machines here, we want to run our finals tests on it, but setting up an Oculus to work on latedays seems infeasible.

The outstanding issues are to actually make the raytracer run at a more acceptable speed, and setting up / dealing with capturing movement data from the Oculus. There's also the potential for misrepresented results, since our final tests will be on a machine that's not actually outputting the images to a screen, but just a file, which might be a nontrivial amount faster than actually displaying the images. So hitting the goal of "acceptable framerate" there might not translate to actually acceptable framerates on actual hardware.

##Updated Schedule
**By 4/22:** Combine the raytracing and VR parts of the project. Additionally, plan out what the potential best ways to improve performance will be, and research more about GPU raytracing. (Both people)

**By 4/26:** Implement the previously outlined CUDA performance boosts, seeing which perform the best (Zach). Additionally, consolidating the workloads, to be something like all the ray-scene intersections are processed at once, and then all of the material processing is done next. (Chris)

**By 4/29:** Research and begin implementing potential VR-specific optimizations, like processing the middle of each eye in more detail. (Chris) Additionally, finish up miscellaneous outstanding CUDA issues, and being writing the testing harness to be run on latedays. (Zach)

**By 5/3:** Finish implementing VR-specific optimizations, and analyze which are actually worthwhile to keep, and do general miscellaneous performance boosts, that arose from our other code. (Chris) Additionally, finish writing the testing harness to be run on latedays, and begin actually running tests on latedays. Time permitting, look into implementing more complex raytracing algorithms. (Zach)

**By 5/9:** Any last minute outstanding performance / correctness issues. (Both) Perform actual testing, and create a way to get the output from latedays to get back onto the Oculus to be used in a demo. Time permitting, implement more complex raytracing algorithms for faster performance.

##Original Project Proposal

##Summary
The goal of our project is to write a fully functional raytracer in CUDA to render photorealistic images to an Oculus in real time. We will take advantage of the parallel processing power of a modern GPU to accelerate the raytracer to the point where it can meet the low latency required of VR applications.

##Background
Raytracing is a rendering technique that works by simulating the way light travels through and interacts with a scene. The simplest form involves tracing a number of rays passing through every pixel of the image and calculating where they intersect with scene geometry, then tracing an additional ray to a light source in the scene to determine if the light source is visible from the intersection point. This information is used to calculate the irradiance at the intersection point based on the properties of the material, represented by a Bidirectional Reflectance Distribution Function (BSDF), which is then used to determine the color of the pixel. A more advanced algorithm will also calculate the effect of indirect lighting from light rays which bounce around the scene a few times. The method we will use, called path tracing, does this by tracing additional rays originating from the scene intersection and performing the same irradiance calculation at a new intersection point, accumulating the result into the irradiance of the initial intersection. This process can be done recursively, in theory allowing arbitrarily long paths. The pseudocode for this algorithm looks like this:

```python
def traceRay(ray):
    x = intersect(ray, scene)
    shadowRay = getRay(x, lightSource)
    irr = 0
    if intersect(ray, scene) in lightSource:
      irr += computeIrradiance(ray, x, shadowRay)
    newRay = sampleRay(x, x.material)
    irr += traceRay(newRay)
    return irr
    
for pixel in image:
  for nSamples iterations:
    ray = sampleRayThroughPixel(pixel)
    pixel.val += traceRay(ray)
```

Other statistical techniques are used to improve the quality of the output with fewer samples, but this doesn't change the basic structure of the algorithm. Most of the parallelism in this algorithm comes from the fact that individual rays can be traced independently, so for a reasonably large image there could millions of rays that could be traced in parallel. The most computationally intensive part of this process is the ray-scene intersection, which involves traversing a structure such as a Bounding Volume Hierarchy or a KD-Tree, so this most likely the part that will require the most optimization for parallel execution.

The issue on the VR side of things is that in order for the scenes to look realistic, they need to be rendered at a high frame rate (typically 60-75 Hz), which requires a high performance machine. The need for such a high frame rate arises because the scene is supposed to move around as the user's head moves. If this isn't rendered fast enough, the position of the camera in the scene will appear as discrete movements, not a continuous and smooth one. This usually leads to user discomfort, and stops the scene from appearing realistic. One of the potential VR-specific optimizations we may be able take advantage of is the possibility of information sharing between the two views. For instance, once we've calculated the indirect lighting at a point in the scene (an expensive operation since it requires tracing a number of additional rays) we may be able to use that information to provide a sample for each eye, essentially getting one sample for free.

##The Challenge
The major challenge of this project is going to be meeting the very low latency requirements of the VR system. Ray tracing in and of itself is a computationally intensive task (assuming you want a high quality output), and speeding that up to the point of getting acceptable frame rate will be a big challenge. In addition to that, there is the challenge of implementing a ray tracer on a GPU. GPUs typically have very wide SIMD units, which we would need to leverage in order to get the best performance. Rays for indirect lighting tend to end up passing through different directions in the scene, which may lead to divergent SIMD execution and incoherrent memory access. We'll have to come up with a method of doing scene intersections that minimizes these issues in order to get good enough performance.

##Resources
We will use the raytracing code we wrote for graphics last semester as a starting point, but as we port it to the GPU we
expect most of it will need to be heavily rewritten. We already have an Oculus to work with, and for most of the project we
will run our code on our own laptops. We will be using https://mediatech.aalto.fi/~samuli/publications/aila2009hpg_paper.pdf and https://mediatech.aalto.fi/~samuli/publications/laine2013hpg_paper.pdf as references to guide the design of our GPU implementation. At the end of the project we would like to run our code on a more powerful GPU than we have in our laptops, but we would need physical access to the machine to use the Oculus, and it needs to run Windows. If we can't get access to such a machine we will run our code on Latedays without rendering to the Oculus so we can still get performance measurements.

##Goals

The primary goal that we plan to achieve is a system capable of rendering images of a reasonable quality to the Oculus in real time. Specifically this means being able to run at 60fps, which means the whole rendering pipeline must have a latency under 16ms. We expect to achieve this performance on relatively simple scenes with potentially complex indirect lighting effects. If we make quick progress on the project, we hope to improve the quality of our output by implementing a more advanced renderer, though we will most likely keep the frame rate at 60fps. At the parallelism competition we will show a demo of a scene being rendered in real time on the Oculus. We will also show results on how quickly our system runs on input with varying complexity.

##Platform
Most of our code will be written in CUDA and run on Nvidia GPUs. Powerful GPUs are already a requirement of doing rasterization for VR, so it makes sense to use the same platform for our raytracer. CUDA is a natural choice since we know we're running on Nvidia hardware and we already have some experience with it.

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

