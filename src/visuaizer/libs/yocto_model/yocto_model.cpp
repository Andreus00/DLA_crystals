//
// Implementation for Yocto/Model
//

//
// LICENSE:
//
// Copyright (c) 2016 -- 2021 Fabio Pellacini
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------

#include "yocto_model.h"

#include <yocto/yocto_sampling.h>

#include <algorithm>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#include "ext/perlin-noise/noise1234.h"
#include "../../../generator/serial/single_core_dla_v2.c"
#include "../../../generator/parallel/parallel_dla_openmp.c"

// -----------------------------------------------------------------------------
// USING DIRECTIVES
// -----------------------------------------------------------------------------
namespace yocto {

// using directives
using std::array;
using std::string;
using std::vector;
using namespace std::string_literals;

}  // namespace yocto

// -----------------------------------------------------------------------------
// IMPLEMENTATION FOR EXAMPLE OF PROCEDURAL MODELING
// -----------------------------------------------------------------------------
namespace yocto {


void init_parameters(struct coords space_size, struct voxel *space, dinamic_list **freezed, struct particle_lists* particles) {
    init_voxel(space, space_size);

    // inizializzazione del cristallo iniziale
    int W = 8;
    struct coords initial_crystal = {W / 2 * 16, W / 2 * 16, W / 2 * 16};
    setValue(space, initial_crystal, -1);

    *freezed = dinamic_list_new();

    particles->list1 = (struct particle **) malloc(sizeof(struct particle *) * PART_NUM);
    particles->freezed = *freezed;
    particles->last1 = PART_NUM -1;
}

void make_grass(scene_data& scene, const instance_data& object,
    const vector<instance_data>& grasses, const grass_params& params) {
     int rng = 69420;
     int mode = 1;
    
    // inizializzazione del volxel

    const int W = 8;

    struct voxel space;
    struct coords space_size = {W, W, W};
    dinamic_list *freezed;
    struct particle_lists particles;
    
    struct timeval tval_before, tval_after, tval_result;    
    if (mode == 0) {
        // inizializzazione delle particelle
        printf("Serial\n");
        gettimeofday(&tval_before, NULL);
        init_parameters(space_size, &space, &freezed, &particles);
        init_particles(particles.list1, PART_NUM, &space);
        
        while(particles.last1 > 0) {
            single_core_dla(&space, &particles);
            
        }
    }
    else {
        printf("Parallel OpenMP - Threads: %d\n", NUM_THREADS);
        gettimeofday(&tval_before, NULL);
        init_parameters(space_size, &space, &freezed, &particles);
        init_particles_parallel(particles.list1, PART_NUM, &space);
        
        while(particles.last1 >= 0){
            parallel_dla_openmp(&space, &particles);      
        }
        
       
    }
    // else if(mode == 2) {
    //     printf("Parallel CUDA\n");
    //     gettimeofday(&tval_before, NULL);
    //     // int *space_linear = parallel_dla_cuda(space_size, CHUNK_SIZE, PART_NUM);
    //     // creo le pipe
    //     int fd[2];  // 0 per leggere, 1 per scrivere
    //     int fd2[2];
    //     if (pipe(fd) == -1) { 
    //         fprintf(stderr, "Pipe Failed");
    //         return;
    //     }
    //     if (pipe(fd2) == -1) {
    //         fprintf(stderr, "Pipe Failed");
    //         return;
    //     }
    //     int pid = 0;
    //     printf("Fork\n");
    //     pid = fork();
    //     if (pid < 0) {
    //         fprintf(stderr, "fork Failed");
    //         return;
    //     }
    //     if(pid == 0) { // figlio
    //         char *const parmList[] = {(char *)  ((long) fd[0]), (char *) ((long) fd[1]), (char *)  ((long) fd2[0]), (char *)  ((long) fd2[1])};
    //         // char argv[4];
    //         // argv[0] = fd[0];    // read for child
    //         // argv[1] = fd[1];    // write for parent
    //         // argv[2] = fd2[0];   // read for parent
    //         // argv[3] = fd2[1];   // write for child
    //         // char *const parmList[] = {"-l", "./../../bin/parallel_dla_cuda"};
    //         //char *a = 
    //         printf("Execl\n");
    //         execv("./parallel_dla_cuda", parmList);
    //         return;
    //     }
    //     else { // padre
    //         close(fd[0]);
    //         close(fd2[1]);
    //         // send data
    //         int ch_sz = CHUNK_SIZE;
    //         int pt_nm = PART_NUM;
    //         ssize_t wrt_num;
    //         printf("Write\n");
    //         wrt_num = write(fd[1], &space_size, sizeof(struct coords));
    //         wrt_num = write(fd[1], &ch_sz, sizeof(int));
    //         wrt_num = write(fd[1], &pt_nm, sizeof(int));
    //         printf("fine scrittura\n");
            
    //         int status;
    //         struct coords voxel_size = {space_size.x * CHUNK_SIZE, space_size.y * CHUNK_SIZE, space_size.z * CHUNK_SIZE};
    //         int space_number = voxel_size.x * voxel_size.y * voxel_size.z;
    //         int *buffer = (int *)calloc(space_number, sizeof(int));
    //         //printf("aspetto lettura\n");
    //         wrt_num = read(fd2[0],buffer ,sizeof(int) * space_number);
    //         printf("building voxel %d\n", (int) wrt_num);
    //         for(int i = 0; i < voxel_size.x; i++) {    // Ho W * W * W chunk
    //             for(int j = 0; j < voxel_size.y; j++) {
    //                 for(int k = 0; k < voxel_size.z; k++) {
    //                     int value = buffer[voxel_size.x * voxel_size.y * k + voxel_size.x * j + i];
    //                     if(value ==-1) printf("x: %d - y: %d - z: %d - val: %d\n", i , j, k, value);
    //                     struct coords c = {i, j, k};
    //                     setValue(&space, c, value);
    //                 }
    //             }
    //         }
    //     }
    // }
    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);
    printf("Time elapsed: %ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
    
    instance_data instance;
    scene.shapes.push_back(make_box());
    int shape_idx = scene.shapes.size() - 1;
    material_data mat_0;
    mat_0.color = vec3f{1, 1, 1};
    mat_0.type = material_type::matte;
    scene.materials.push_back(mat_0);
    material_data mat_1;
    mat_1.color = vec3f{0.7, 0.7, 0.7};
    mat_1.type = material_type::matte;
    scene.materials.push_back(mat_1);
    material_data mat_2;
    mat_2.color = vec3f{0.5, 0.5, 0.5};
    mat_2.type = material_type::matte;
    scene.materials.push_back(mat_2);
    int material_idx = scene.materials.size() - 3;
    struct coords s = getSize(&space);
    int size = s.x * s.y * s.z;
    for(int i = 0; i < W; i++) {    // Ho W * W * W chunk
        for(int j = 0; j < W; j++) {
            for(int k = 0; k < W; k++) {
                int index = W * W * i + W * j + k;
                struct chunk c = space.chunks[index];
                
                for(int x = 0 ; x < CHUNK_SIZE; x++) {
                    for(int y = 0 ; y < CHUNK_SIZE; y++) {
                        for(int z = 0 ; z < CHUNK_SIZE; z++) {
                            auto cell = c.blocks[z][y][x];
                            //std::cout << i << "i "<< x << "x "<< y << "y "<< z << "z\n";
                            if (cell == -1){
                                instance.shape = shape_idx;
                                vec3f box = vec3f{float( x + CHUNK_SIZE * k - W * CHUNK_SIZE/ 2 ),float( y + CHUNK_SIZE * j - W / 2 * CHUNK_SIZE),float(z + CHUNK_SIZE * i - W / 2 * CHUNK_SIZE)};
                                instance.frame.o=vec3f{box.x,box.y,box.z};
                                instance.material = material_idx + (fmod(distance(box, vec3f{0, 0, 0}), 3));
                                scene.instances.push_back(instance);

                            }
                        }
                    }
                }
            }
        }
    }
    
}
/////////////////////////////////////////////////

}  // namespace yocto

