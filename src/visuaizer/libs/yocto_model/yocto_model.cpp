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

#include "ext/perlin-noise/noise1234.h"
#include "../../../generator/serial/single_core_dla.c"
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

void make_grass(scene_data& scene, const instance_data& object,
    const vector<instance_data>& grasses, const grass_params& params) {
     int rng = 69420;
    
    // inizializzazione del volxel

    const int W = 8;

    struct voxel space;
    struct coords space_size = {W, W, W};
    init_voxel(&space, space_size);


    // inizializzazione del cristallo iniziale

    struct coords initial_crystal = {W / 2 * 16, W / 2 * 16, W / 2 * 16};
    setValue(&space, initial_crystal, -1);

    // inizializzazione delle particelle

    dinamic_list *particle_list = dinamic_list_new();

    // inizializzazione della lista delle particelle da cristallizzare

    dinamic_list *freezed = dinamic_list_new();
    
    init_particles(particle_list, PART_NUM, &rng, &space);

    while(particle_list->last >= 0) {
        single_core_dla(&space, particle_list, freezed, &rng);
    }
    
    instance_data instance;
    scene.shapes.push_back(make_box());
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
                                instance.shape=scene.shapes.size()-1;
                                instance.material= 0;
                                instance.frame.o=vec3f{float( x + CHUNK_SIZE * k - W / 2 * CHUNK_SIZE),float( y + CHUNK_SIZE * j - W / 2 * CHUNK_SIZE),float(z + CHUNK_SIZE * i - W / 2 * CHUNK_SIZE)};

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

