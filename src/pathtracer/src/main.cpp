#include "CMU462/CMU462.h"
#include "CMU462/viewer.h"

#define TINYEXR_IMPLEMENTATION
#include "CMU462/tinyexr.h"

#include "application.h"

#include <iostream>
#include <unistd.h>

using namespace std;
using namespace CMU462;

#define msg(s) cerr << "[PathTracer] " << s << endl;

void usage(const char* binaryName) {
  printf("Usage: %s [options] <scenefile>\n", binaryName);
  printf("Program Options:\n");
  printf("  -s  <INT>        Number of camera rays per pixel\n");
  printf("  -l  <INT>        Number of samples per area light\n");
  printf("  -t  <INT>        Number of render threads\n");
  printf("  -m  <INT>        Maximum ray depth\n");
  printf("  -e  <PATH>       Path to environment map\n");
  printf("  -b  <INT>        Number of rays to run benchmark on\n");
  printf("  -h               Print this help message\n");
  printf("\n");
}

/*
int main( int argc, char** argv ) {

  // get the options
  AppConfig config; int opt;
  bool benchmark = false;
  int numRays = 0;
  while ( (opt = getopt(argc, argv, "s:l:t:m:e:h:b:")) != -1 ) {  // for each option...
    switch ( opt ) {
    case 's':
        config.pathtracer_ns_aa = atoi(optarg);
        break;
    case 'l':
        config.pathtracer_ns_area_light = atoi(optarg);
        break;
    case 't':
        config.pathtracer_num_threads = atoi(optarg);
        break;
    case 'm':
        config.pathtracer_max_ray_depth = atoi(optarg);
        break;
    case 'b':
        benchmark = true;
        numRays = atoi(optarg);
        break;
    default:
        usage(argv[0]);
        return 1;
    }
  }

  // print usage if no argument given
  if (optind >= argc) {
    usage(argv[0]);
    return 1;
  }

  string sceneFilePath = argv[optind];
  msg("Input scene file: " << sceneFilePath);

  // parse scene
  Collada::SceneInfo *sceneInfo = new Collada::SceneInfo();
  if (Collada::ColladaParser::load(sceneFilePath.c_str(), sceneInfo) < 0) {
    delete sceneInfo;
    exit(0);
  }

  // create application
  Application app (config);

  // load scene
  app.load(sceneInfo);

  delete sceneInfo;

  if (benchmark) app.benchmark(numRays);
  else app.pathtrace();

  exit(EXIT_SUCCESS); // shamelessly faking it

  return 0;

}*/


