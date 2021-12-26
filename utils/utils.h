#include <string>
#include <math.h>

#define CHUNK_SIZE 16
// size of voxels in chunks
struct coords {
    int x, y, z;
}

// a chunk of the voxel
struct chunk {
    int blocks[CHUNK_SIZE][CHUNK_SIZE][CHUNK_SIZE];
}

// the entire voxel space
struct voxel {
    chunk* chunks;  // List of chunks. Treated like a 3D array
    coords c;         // Size of the chunks
}

// initialize a voxel
void init_voxel(voxel* v, coords c) {
    v->chunks = (chunk*) malloc(c->x * c->y * c->z, sizeof(chunk));
    v->c.x = c.x;
    v->c.y = c.y;
    v->c.z = c.z;
}

// get a block given the block's coordinates
chunk* getBlock(voxel* v, coords c) {
    if (c.x > v->c.x || c.y > v->c.y || c.z > v->c.z) return NULL;
                    // (        z         )   (     y     )  ( x )
    return &v->chunks[v->c.x * v->c.y * c->z + v->c.x * c.y + c.x];
}

// get the value of a cell
int getValue(voxel* v, coords c) {
    // recupero il blocco
    int x_block = c.x / CHUNK_SIZE;
    int y_block = c.y / CHUNK_SIZE;
    int z_block = c.z / CHUNK_SIZE;

    chunk* b = getBlock(v, struct coords{x_block, y_block, z_block});

    int x = c.x % CHUNK_SIZE;
    int y = c.y % CHUNK_SIZE;
    int z = c.z % CHUNK_SIZE;

    return b[CHUNK_SIZE * CHUNK_SIZE * z + CHUNK_SIZE * y + x];
}

// set the value of a cell
int setValue(voxel* v, coords c, int value) {
    // recupero il blocco
    int x_block = c.x / CHUNK_SIZE;
    int y_block = c.y / CHUNK_SIZE;
    int z_block = c.z / CHUNK_SIZE;

    chunk* b = getBlock(v, struct coords{x_block, y_block, z_block});

    int x = c.x % CHUNK_SIZE;
    int y = c.y % CHUNK_SIZE;
    int z = c.z % CHUNK_SIZE;

    *(b[CHUNK_SIZE * CHUNK_SIZE * z + CHUNK_SIZE * y + x]) = value;
}