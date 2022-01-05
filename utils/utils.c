#include "utils.h"
#include <malloc.h>
#include <omp.h>
// initialize a voxel
void init_voxel(struct voxel* v, struct coords c) {
    v->chunks = (struct chunk *) calloc(c.x * c.y * c.z, sizeof(struct chunk));
    v->c.x = c.x;
    v->c.y = c.y;
    v->c.z = c.z;
}

// get a chunk given the chunk's coordinates
struct chunk* getChunk(struct voxel* v, struct coords c) {
    if (c.x >= v->c.x || c.y >= v->c.y || c.z >= v->c.z) return NULL;
                    // (        z         )   (     y     )  ( x )
    return &v->chunks[v->c.x * v->c.y * c.z + v->c.x * c.y + c.x];
}

// get the value of a cell
int getValue(struct voxel* v, struct coords c, int* out) {
    // recupero il blocco
    int x_block = c.x / CHUNK_SIZE;
    int y_block = c.y / CHUNK_SIZE;
    int z_block = c.z / CHUNK_SIZE;
    struct coords coord = {x_block, y_block, z_block};
    struct chunk* b = getChunk(v, coord);

    if(b == NULL) return -1;

    int x = c.x % CHUNK_SIZE;
    int y = c.y % CHUNK_SIZE;
    int z = c.z % CHUNK_SIZE;

    *out = b->blocks[z][y][x];
    return 0;
}

// set the value of a cell
int setValue(struct voxel* v, struct coords c, int value) {
    // recupero il blocco
    int x_block = c.x / CHUNK_SIZE;
    int y_block = c.y / CHUNK_SIZE;
    int z_block = c.z / CHUNK_SIZE;
    struct coords coord = {x_block, y_block, z_block};
    struct  chunk* b = getChunk(v, coord);
    if (b == NULL) {
        return -1;
    }

    int x = c.x % CHUNK_SIZE;
    int y = c.y % CHUNK_SIZE;
    int z = c.z % CHUNK_SIZE;

    b->blocks[z][y][x] = value;
    return 0;
}

// set the value of a cell
int incrementValue(struct voxel* v, struct coords c) {
    // recupero il blocco
    int x_block = c.x / CHUNK_SIZE;
    int y_block = c.y / CHUNK_SIZE;
    int z_block = c.z / CHUNK_SIZE;
    struct coords coord = {x_block, y_block, z_block};
    struct  chunk* b = getChunk(v, coord);
    if (b == NULL) {
        return -1;
    }

    int x = c.x % CHUNK_SIZE;
    int y = c.y % CHUNK_SIZE;
    int z = c.z % CHUNK_SIZE;

    b->blocks[z][y][x]++;
    return 0;
}

// set the value of a cell
int decrementValue(struct voxel* v, struct coords c) {
    // recupero il blocco
    int x_block = c.x / CHUNK_SIZE;
    int y_block = c.y / CHUNK_SIZE;
    int z_block = c.z / CHUNK_SIZE;
    struct coords coord = {x_block, y_block, z_block};
    struct  chunk* b = getChunk(v, coord);
    if (b == NULL) {
        return -1;
    }

    int x = c.x % CHUNK_SIZE;
    int y = c.y % CHUNK_SIZE;
    int z = c.z % CHUNK_SIZE;

    b->blocks[z][y][x]--;
    return 0;
}


struct coords getSize(struct voxel* v) {
    struct coords ret;
    ret.x = v->c.x * CHUNK_SIZE;
    ret.y = v->c.y * CHUNK_SIZE;
    ret.z = v->c.z * CHUNK_SIZE;
    return ret;
}

////////////////////////
// pseudo random function
float random_float(int* rng) {
    int n  = (*rng << 13U) ^ *rng;
    (*rng)++;
    n = n * (n * n * 15731U + 789221U) + 1376312589U;
    return (float)((n & 0x7fffffffU)/((float)(0x7fffffff)));
}

////////////////////////
// pseudo random function
float atomic_random_float(int* rng) {
    int n  = (*rng << 13U) ^ *rng;
    (*rng) *= 6364136223846793005ULL;
    // #pragma omp parallel for
    // for(int i = 0 ; i < 100; i++) {
    //     printf("%d ", i);
    // }
    // printf("\n");
    n = n * (n * n * 15731U + 789221U) + 1376312589U;

    return (float)((n & 0x7fffffffU)/((float)(0x7fffffff)));
}

