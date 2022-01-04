#include "../src/generator/parallel/parallel_dla_openmp.c"
#include "../src/generator/serial/single_core_dla_v2.c"
#include <sys/time.h>


int main(int argc, char const *argv[])
{
    // int rng = 69420;
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
    struct timeval tval_before, tval_after, tval_result;
    struct particle_lists particles;
    particles.list1= (struct particle **) malloc(sizeof(struct particle *) * PART_NUM);
    particles.freezed= freezed;
    particles.last1= PART_NUM -1;
    if (SERIAL) {
        // inizializzazione delle particelle
        printf("Serial\n");
       
        init_particles(particles.list1, PART_NUM, &space);
        
        while(particles.last1 >= 0) {
            gettimeofday(&tval_before, NULL);
            single_core_dla(&space, &particles);
            gettimeofday(&tval_after, NULL);
            timersub(&tval_after, &tval_before, &tval_result);
            printf("Time elapsed: %ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
            return 0;
        }
    }
    else {
        printf("Parallel - Threads: %d\n", NUM_THREADS);
        
        init_particles_parallel(particles.list1, PART_NUM, &space);
        while(particles.last1 >= 0){
            gettimeofday(&tval_before, NULL);
            parallel_dla_openmp(&space, &particles);
            gettimeofday(&tval_after, NULL);
            timersub(&tval_after, &tval_before, &tval_result);
            printf("Time elapsed: %ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
            return 0;
        }
    }
    
    return 0;
}
