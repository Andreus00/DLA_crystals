#include "../src/generator/parallel/parallel_dla_openmp.c"
#include "../src/generator/serial/single_core_dla_v2.c"
#include <sys/time.h>


int main(int argc, char const *argv[])
{
    if (argc != 7){
        fprintf(stderr, "Usage: ./paralle_test_cuda part_num voxel_size num_threads [serial? 0 - 1] [print time? 0 - 1] [print crystal? 0 - 1]\n");
        exit(1);
    }
    
    
    // inizializzazione del volxel
    int const NUM_THREADS = atoi(argv[3]);
    const int PART_NUM = atoi(argv[1]);
    const int SERIAL = atoi(argv[4]);
    const int W = atoi(argv[2]);
    struct voxel space;
    struct coords space_size = {W, W, W};
    init_voxel(&space, space_size);


    // inizializzazione del cristallo iniziale

    struct coords initial_crystal = {W / 2 * CHUNK_SIZE, W / 2 * CHUNK_SIZE, W / 2 * CHUNK_SIZE};
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
        gettimeofday(&tval_before, NULL);
        
        while(particles.last1 >= 0) {
            single_core_dla(&space, &particles);
            
        }
        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);
        if (atoi(argv[5])){
            printf("Time elapsed: %ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
        }
        
    }
    else {
        printf("Parallel - Threads: %d\n", NUM_THREADS);
        
        init_particles_parallel(particles.list1, PART_NUM, &space, NUM_THREADS);
        
        gettimeofday(&tval_before, NULL);
        while(particles.last1 >= 0){
            parallel_dla_openmp(&space, &particles, NUM_THREADS);
        }
        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);
        if (atoi(argv[5])){
            printf("Time elapsed: %ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
        }
              
         
    }
    struct coords voxel_size = {space_size.x * CHUNK_SIZE, space_size.y * CHUNK_SIZE, space_size.z * CHUNK_SIZE};
    if (atoi(argv[6])){
        for(int i = 0; i < voxel_size.x; i++) {  
            for(int j = 0; j < voxel_size.y; j++) {
                for(int k = 0; k < voxel_size.z; k++) {
                    struct coords c = {i, j, k};
                    int value;
                    getValue(&space, c, &value);
                    printf("x: %d - y: %d - z: %d - val: %d\n", i, j, k, value);
                }
            }
         }
    }
    
    return 0;
}
