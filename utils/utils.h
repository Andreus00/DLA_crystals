#include <math.h>

#ifndef DINAMIC_LISTS
    #include "dinamic_list.h"
#endif

#define UTILS

#define CHUNK_SIZE 16

// size of voxels in chunks
struct coords {
    int x, y, z;
};

struct cell_info {
    struct coords coord;
    int particles;
};

// a chunk of the voxel
struct chunk {
    int blocks[CHUNK_SIZE][CHUNK_SIZE][CHUNK_SIZE];
};

// the entire voxel space
struct voxel {
    struct chunk* chunks;  // List of chunks. Treated like a 3D array
    struct coords c;         // Size of the chunks
};

struct particle {
    struct coords coord;
    int rng;
};

struct particle_lists {
    struct particle **list1;
    dinamic_list *freezed;
    int last1;
};

// initialize a voxel
void init_voxel(struct voxel* v, struct coords c);

// get a chunk given the chunk's coordinates
struct chunk* getChunk(struct voxel* v, struct coords c);

// get the value of a cell
int getValue(struct voxel* v, struct coords c, int* out);

// set the value of a cell
int setValue(struct voxel* v, struct coords c, int value);

struct coords getSize(struct voxel* v);



//////////////////////////////
// random generator

float random_float(int *rng);

float atomic_random_float(int *rng);

