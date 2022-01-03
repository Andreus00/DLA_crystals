#include "../src/generator/parallel/parallel_dla_openmp.c"
#include "../src/generator/serial/single_core_dla.c"



int main(int argc, char const *argv[])
{
    int rng = 69420;
     int const SERIAL = 0;
    
    // inizializzazione del volxel

    const int W = 8;

    struct voxel space;
    struct coords space_size = {W, W, W};
    init_voxel(&space, space_size);


    // inizializzazione del cristallo iniziale

    struct coords initial_crystal = {W / 2 * 16, W / 2 * 16, W / 2 * 16};
    setValue(&space, initial_crystal, -1);


    // inizializzazione della lista delle particelle da cristallizzare

    dinamic_list *freezed = dinamic_list_new();
    
    if (SERIAL) {
        // inizializzazione delle particelle

        dinamic_list *particle_list = dinamic_list_new();
        init_particles(particle_list, PART_NUM, &rng, &space);
        while(particle_list->last >= 0) {
            single_core_dla(&space, particle_list, freezed, &rng);
        }
    }
    else {
        struct particle_lists particles;
        particles.list1= (struct coords **) malloc(sizeof(struct coords *) * PART_NUM);
        particles.list2= (struct coords **) malloc(sizeof(struct coords *) * PART_NUM);
        particles.freezed= freezed;
        particles.last1= PART_NUM -1;
        particles.last2= -1;
        // for (int i = 0; i < 1000; i++)
        //     printf("aaa");
        init_particles_parallel(particles.list1, PART_NUM, &rng, &space);
        
        while(particles.last1 >= 0){
            
            parallel_dla_openmp(&space, &particles, &rng);
            particles.last1 = particles.last2;
            particles.last2 = -1;
            struct coords **p = particles.list1;
            particles.list1 = particles.list2;
            particles.list2 = p;
        }
    }
    return 0;
}
