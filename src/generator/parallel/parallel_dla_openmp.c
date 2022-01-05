
#ifndef UTILS
    #include "../../../utils/utils.c"
#endif
#ifndef DINAMIC_LISTS
    #include "../../../utils/dinamic_list.h"
#endif
#include <omp.h>
#define PART_NUM 10000000
#define NUM_THREADS 8

struct particle *get_random_position_atomic(struct coords voxel_size, int seed) {
    struct particle *ret = (struct particle *) malloc(sizeof(struct particle));
    ret->rng = seed;
    ret->coord.x = (int) (atomic_random_float(&ret->rng) * voxel_size.x);
    ret->coord.y = (int) (atomic_random_float(&ret->rng) * voxel_size.y);
    ret->coord.z = (int) (atomic_random_float(&ret->rng) * voxel_size.z);
    switch((int) atomic_random_float(&ret->rng) * 6) {
        case 0:
            ret->coord.x = 0;
            break;
        case 1:
            ret->coord.x = voxel_size.x - 1;
            break;
        case 2:
            ret->coord.y = 0;
            break;
        case 3:
            ret->coord.y = voxel_size.y - 1;
            break;
        case 4:
            ret->coord.z = 0;
            break;
        default: 
            ret->coord.z = voxel_size.z - 1;
            break;
    }
    return ret;
    
}
void move_particle_atomic(struct coords *c1, struct coords *out, int* rng, struct coords voxel_size){
    out->x = c1->x;
    out->y = c1->y;
    out->z = c1->z;
    int rx = ((int) (atomic_random_float(rng) * 3)) - 1;
    int ry = ((int) (atomic_random_float(rng) * 3)) - 1;
    int rz = ((int) (atomic_random_float(rng) * 3)) - 1;
    if((rx + out->x ) > 0 &&  (rx + out->x ) < voxel_size.x)
        out->x += rx;
    if((ry + out->y ) > 0 &&  (ry + out->y ) < voxel_size.y)
        out->y += ry;
    if((rz + out->z ) > 0 &&  (rz + out->z ) < voxel_size.z)
        out->z += rz;
    return;
}

void init_particles_parallel(struct particle **list, int num, struct voxel *v) {
    struct coords size = getSize(v);
    #pragma omp parallel for  num_threads(NUM_THREADS)
    for (int i=0; i<num; i++) {
        list[i] = get_random_position_atomic(size, i + 1);
    } 
}

void parallel_dla_openmp(struct voxel *space, struct particle_lists *particles) { 
    omp_set_num_threads(NUM_THREADS);
    
    struct coords voxel_size = getSize(space);
    // per ogni particella
    
    // int rng_ = *rng;
    // atomic_random_float(rng);
    #pragma omp parallel
{
    #pragma omp for schedule(dynamic, 1000)// firstprivate(rng_)
        for (int i = 0; i <= particles->last1; i++) {
            // valore presente in una cella del voxel
            int cell_value;
            struct particle *part = (struct particle *)particles->list1[i];
            // controlla se la particella che si vuole muovere è stata cristallizzata
            getValue(space, part->coord, &cell_value);
            // particella  non cristallizzata
            if(cell_value != -1) {
                // calcolo nuova posizione della particella
                struct coords new_position;
                move_particle_atomic(&(part->coord), &new_position, &(part->rng), voxel_size);
                // controllo se la particella si sta spostando verso un cristallo
                getValue(space, new_position, &cell_value);
                //particella da cristallizzare
                if(cell_value ==-1){
                    // aggiunge alla lista di particelle da cristallizzare
                    dinamic_list_add(particles->freezed, part);
                    particles->list1[i] = NULL;
                }
                // si muove verso uno spazio vuoto
                else{
                    // aggiorna la posizione della particella con la nuova posizione
                    
                    part->coord = new_position;
                }
            }
            // particella cristallizzata
            else{
                free(particles->list1[i]);
                particles->list1[i] = NULL;
            }
        }
    
    
    //printf("%d, %d\n", particles->last1, *rng);
    #pragma omp barrier

    //      cristallizza le particelle che si trovano vicino a un cristallo
    // ciclo sulle particelle che devono essere cristallizzate
    #pragma omp for
    for(int i = 0; i  <= particles->freezed->last; i++) {
        // recupero l'elemento dalla lista dei freezed
        // element e = dinamic_list_pop_last(particles->freezed);
        // controllo se ci sono stati errori
        // if(e.error != 0) {
        //     continue;
        // }
        // setto il valore della cella a -1
        setValue(space, ((struct particle *) particles->freezed->list[i])->coord, -1);
        if(particles->last1 % ((int)(PART_NUM / 10)) == 0)
            printf("%d / %d\n", particles->last1, PART_NUM);
        free(particles->freezed->list[i]);
    }
    #pragma omp single
    {
        int last = -1;
        for(int i = 0; i <= particles->last1; i++){
            if (particles->list1[i] != NULL)
                 particles->list1[++last] = particles->list1[i];
        }
        particles->last1 = last;
    }
}
    dinamic_list_clear(particles->freezed);
    
    

}