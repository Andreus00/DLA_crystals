//
// LICENSE:
//
// Copyright (c) 2016 -- 2020 Fabio Pellacini
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#include <yocto/yocto_cli.h>
#include <yocto/yocto_scene.h>
#include <yocto/yocto_sceneio.h>
#include <yocto_model/yocto_model.h>
using namespace yocto;

#include <filesystem>
#include <iostream>

instance_data& get_instance(scene_data& scene, const string& name) {
  for (auto idx = 0; idx < scene.instances.size(); idx++) {
    if (scene.instance_names[idx] == name) return scene.instances[idx];
  }
  throw std::out_of_range{"unknown instance " + name};
}

void run(const vector<string>& args) {
  // command line parameters
  auto terrain            = ""s;
  auto tparams            = terrain_params{};
  auto displacement       = ""s;
  auto dparams            = displacement_params{};
  auto hair               = ""s;
  auto hairbase           = ""s;
  auto hparams            = hair_params{};
  auto grass              = ""s;
  auto grassbase          = ""s;
  auto gparams            = grass_params{};
  auto output             = "out.json"s;
  auto filename           = "scene.json"s;
  auto dense_hair         = false;
  auto sample_elimination = false;
  auto world              = false;
  auto cell               = false;
  auto smooth_vor         = false;
  auto voronoise_u        = -1.0f;
  auto voronoise_v        = -1.0f;
  auto influence_radius   = 0.005f;
  auto cell_size          = 0.005f;
  auto tree               = false;
  auto trparams           = tree_params{};
  auto woods              = 0;

  // parse command line
  auto error = string{};
  auto cli   = make_cli("ymodel", "Make procedural scenes");
  add_option(cli, "terrain", terrain, "terrain object");
  add_option(cli, "displacement", displacement, "displacement object");
  add_option(cli, "hair", hair, "hair object");
  add_option(cli, "hairbase", hairbase, "hairbase object");
  add_option(cli, "grass", grass, "grass object");
  add_option(cli, "grassbase", grassbase, "grassbase object");
  add_option(cli, "hairnum", hparams.num, "hair number");
  add_option(cli, "hairlen", hparams.lenght, "hair length");
  add_option(cli, "hairstr", hparams.strength, "hair strength");
  add_option(cli, "hairgrav", hparams.gravity, "hair gravity");
  add_option(cli, "hairstep", hparams.steps, "hair steps");
  add_option(cli, "output", output, "output scene");
  add_option(cli, "scene", filename, "input scene");
  add_option(cli, "dense_hair", dense_hair, "dense_hair choice");
  add_option(cli, "world", world, "world generator choice");
  add_option(cli, "cell", cell, "cell noise choice");
  add_option(cli, "smooth_vor", smooth_vor, "smooth voronoise choice");
  add_option(cli, "voronoise_u", voronoise_u, "voronoise_v value");
  add_option(cli, "voronoise_v", voronoise_v, "voronoise_u value");
  add_option(cli, "sample_elimination", sample_elimination,
      "sample_elimination for hair");
  add_option(cli, "influence_radius", influence_radius,
      "influence_radius for sample elimination");
  add_option(cli, "cell_size", cell_size, "cell_size for sample elimination");
  add_option(cli, "tree", tree, "tree");
  add_option(cli, "brsteps", trparams.steps, "number of steps");
  add_option(cli, "step_len", trparams.step_len, "Step len");
  add_option(cli, "range", trparams.range, "Range");
  add_option(cli, "kill_range", trparams.kill_range, "Kill range");
  add_option(
      cli, "crown_radius", trparams.crown_radius, "Randius of the crown");
  add_option(cli, "crown_height", trparams.crown_height,
      "Height of the crown of the tree");
  add_option(cli, "crown_points_distance", trparams.crown_points_distance,
      "Distance used for the sample elimination of the attraction points");
  add_option(cli, "crown_points_num", trparams.crown_points_num,
      "Number of attraction points of the tree's crown");
  add_option(cli, "steps", trparams.steps, "Number of steps");
  add_option(cli, "fork_chance", trparams.fork_chance,
      "chance of a branch to fork when one or more attraction poits are deleted.");
  add_option(cli, "thickness", trparams.thickness, "thickness of the branch");
  add_option(cli, "main_thickness_decrease", trparams.main_thickness_decrease,
      "Decrease of the main branch");
  add_option(cli, "division_thickness_decrease",
      trparams.division_thickness_decrease,
      "Decrease of the thickness when the branch is divided");
  add_option(cli, "ignore_points_behind", trparams.ignore_points_behind,
      "Value from -1 to 1. it affects when an attraction point is ignored, based on his direction.");
  add_option(cli, "branch_strictness", trparams.branch_strictness,
      "This parameter modifies how much the direction of the branch parent affects the direction of the branch son");
  add_option(
      cli, "gravity", trparams.gravity, "Gravity that influences the tree");
  add_option(cli, "show_crown_points", trparams.show_crown_points,
      "Show the attraction points");
  add_option(cli, "woods", woods, "make woods");
  if (!parse_cli(cli, args, error)) print_fatal(error);

  // load scene
  auto scene = scene_data{};
  if (!load_scene(filename, scene, error)) print_fatal(error);

  // set influence_radius and cell_size
  if (influence_radius != 0.005) hparams.influence_radius = influence_radius;

  if (cell_size != 0.005) hparams.cell_size = cell_size;
  // create procedural geometry
  if (woods) {
    make_woods(scene, get_instance(scene, grassbase), woods);
  }
  if (tree) {
    generate_tree(scene, {0, 0, 0},
        {
            0,
            1,
            0,
        },
        trparams);
  }
  if (terrain != "") {
    make_terrain(scene.shapes[get_instance(scene, terrain).shape], tparams);
  }
  if (displacement != "") {
    std::cout << world << cell << smooth_vor << voronoise_u << voronoise_v
              << std::endl;
    if (world) {
      make_world(
          scene.shapes[get_instance(scene, displacement).shape], dparams);
    } else if (cell) {
      make_cell_voro_displacement(
          scene.shapes[get_instance(scene, displacement).shape], dparams);
    } else if (smooth_vor) {
      make_smooth_voro_displacement(
          scene.shapes[get_instance(scene, displacement).shape], dparams);
    } else if (voronoise_u >= 0 && voronoise_v >= 0) {
      make_voro_displacement(
          scene.shapes[get_instance(scene, displacement).shape], dparams,
          min(voronoise_u, 1.0f), min(voronoise_v, 1.0f));
    } else {
      make_displacement(
          scene.shapes[get_instance(scene, displacement).shape], dparams);
    }
  }
  if (hair != "" && !dense_hair) {
    scene.shapes[get_instance(scene, hair).shape]      = {};
    scene.shape_names[get_instance(scene, hair).shape] = "hair";
    if (sample_elimination) {
      make_hair_sample_elimination(
          scene.shapes[get_instance(scene, hair).shape],
          scene.shapes[get_instance(scene, hairbase).shape], hparams);
    } else {
      make_hair(scene.shapes[get_instance(scene, hair).shape],
          scene.shapes[get_instance(scene, hairbase).shape], hparams);
    }
  }
  if (grass != "") {
    auto grasses = vector<instance_data>{};
    for (auto idx = 0; idx < scene.instances.size(); idx++) {
      if (scene.instance_names[idx].find(grass) != string::npos)
        grasses.push_back(scene.instances[idx]);
    }
    make_grass(scene, get_instance(scene, grassbase), grasses, gparams);
  }
  if (hair != "" && dense_hair) {
    scene.shapes[get_instance(scene, hair).shape]      = {};
    scene.shape_names[get_instance(scene, hair).shape] = "hair";
    make_dense_hair(scene, scene.shapes[get_instance(scene, hair).shape],
        get_instance(scene, hairbase), hparams);
  }

  // make a directory if needed
  if (!make_scene_directories(output, scene, error)) print_fatal(error);

  // save scene
  if (!save_scene(output, scene, error)) print_fatal(error);
}

int main(int argc, const char* argv[]) {
  handle_errors(run, make_cli_args(argc, argv));
}
