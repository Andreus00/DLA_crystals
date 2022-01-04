//
// # Yocto/Model: Examples of procedural modeling
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
//

#ifndef _YOCTO_MODEL_H_
#define _YOCTO_MODEL_H_

// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------

#include <yocto/yocto_geometry.h>
#include <yocto/yocto_math.h>
#include <yocto/yocto_scene.h>

#include <array>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// USING DIRECTIVES
// -----------------------------------------------------------------------------
namespace yocto {

// using directives
using std::array;
using std::string;
using std::vector;

}  // namespace yocto

// -----------------------------------------------------------------------------
// EXAMPLE OF PROCEDURAL MODELING
// -----------------------------------------------------------------------------
namespace yocto {

struct terrain_params {
  float size    = 0.1f;
  vec3f center  = zero3f;
  float height  = 0.1f;
  float scale   = 10;
  int   octaves = 8;
  vec4f bottom  = srgb_to_rgb(vec4f{154, 205, 50, 255} / 255);
  vec4f middle  = srgb_to_rgb(vec4f{205, 133, 63, 255} / 255);
  vec4f top     = srgb_to_rgb(vec4f{240, 255, 255, 255} / 255);
};

void make_terrain(shape_data& shape, const terrain_params& params);

struct displacement_params {
  float height  = 0.02f;
  float scale   = 50;
  int   octaves = 8;
  vec4f bottom  = srgb_to_rgb(vec4f{64, 224, 208, 255} / 255);
  vec4f top     = srgb_to_rgb(vec4f{244, 164, 96, 255} / 255);
};

void make_displacement(shape_data& shape, const displacement_params& params);

struct hair_params {
  int   num              = 100000;
  int   steps            = 1;
  float lenght           = 0.02f;
  float scale            = 250;
  float strength         = 0.01f;
  float gravity          = 0.0f;
  vec4f bottom           = srgb_to_rgb(vec4f{25, 25, 25, 255} / 255);
  vec4f top              = srgb_to_rgb(vec4f{244, 164, 96, 255} / 255);
  float influence_radius = 0.005;
  float cell_size        = 0.005;
};

void make_hair(
    shape_data& hair, const shape_data& shape, const hair_params& params);

struct grass_params {
  int num = 10000;
};

void make_grass(scene_data& scene, const instance_data& object,
    const vector<instance_data>& grasses, const grass_params& params);

// extra credit

void make_dense_hair(scene_data& scene, shape_data& hair,
    const instance_data& object, const hair_params& params);
void make_world(shape_data& shape, const displacement_params& params);
void make_cell_voro_displacement(
    shape_data& shape, const displacement_params& params);
void make_smooth_voro_displacement(
    shape_data& shape, const displacement_params& params);
void make_voro_displacement(
    shape_data& shape, const displacement_params& params, float u, float v);
void make_hair_sample_elimination(
    shape_data& hair, const shape_data& shape, const hair_params& params);

struct Branch {
  vec3f start;

  vec3f end;

  vec3f direction;

  int parent_index;

  float thickness;

  vector<int> _children;

  vector<int> _attractors;
};

void          addChild(struct Branch* parent, int child);
vector<int>   getChildren(struct Branch* parent);
vector<vec3f> getAttractors(struct Branch* parent);

void init_branch(
    struct Branch* b, vec3f start, vec3f end, vec3f direction, int parent);

struct tree_params {
  float step_len                    = 0.02;  // len of each step of the segments
  float range                       = 0.1f;  // attraction range
  float kill_range                  = 0.08f;
  float crown_radius                = 0.4f;  // radius of the crown
  float crown_height                = 0.8f;  // height of the crown
  float crown_points_distance       = 0.1;
  float crown_points_num            = 600;  // number of points of the crown
  int   steps                       = 300;
  float fork_chance                 = 0.4f;
  float thickness                   = 0.03;
  float main_thickness_decrease     = 0.99;
  float division_thickness_decrease = 0.75;
  float ignore_points_behind        = -0.5;
  float branch_strictness           = 1.0;
  float gravity                     = 0.0;
  bool  show_crown_points           = false;
  bool  show_range                  = false;
};

void generate_tree(scene_data& scene, const vec3f start, const vec3f norm,
    const tree_params& params);
void make_woods(
    scene_data& scene, const instance_data& object, const int tree_num);
}  // namespace yocto

#endif
