
#ifndef UTILS
    #include "../../../utils/utils.c"
#endif
#ifndef DINAMIC_LISTS
    #include "../../../utils/dinamic_list.h"
#endif
#include <omp.h>
#define PART_NUM 8000
#define NUM_THREADS 4

struct coords *get_random_position_atomic(struct coords voxel_size, int* rng) {
    struct coords *ret = (struct coords *) malloc(sizeof(struct coords));
    ret->x = (int) (atomic_random_float(rng) * voxel_size.x);
    ret->y = (int) (atomic_random_float(rng) * voxel_size.y);
    ret->z = (int) (atomic_random_float(rng) * voxel_size.z);
    switch((int) atomic_random_float(rng) * 6) {
        case 0:
            ret->x = 0;
            break;
        case 1:
            ret->x = voxel_size.x - 1;
            break;
        case 2:
            ret->y =0;
            break;
        case 3:
            ret->y = voxel_size.y - 1;
            break;
        case 4:
            ret->z = 0;
            break;
        default: 
            ret->z = voxel_size.z - 1;
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

void init_particles_parallel(struct coords **list, int num, int* rng, struct voxel *v) {
    struct coords size = getSize(v);
    #pragma omp parallel for schedule(static, 30) num_threads(NUM_THREADS)
    for (int i=0; i<num; i++) {
        list[i] = get_random_position_atomic(size, rng);
    }
}

void parallel_dla_openmp(struct voxel *space, struct particle_lists *particles, int *rng) { 
    omp_set_num_threads(NUM_THREADS);
    
    // per ogni particella
    #pragma omp parallel for schedule(static, 30)
    for (int i = 0; i <= particles->last1; i++) {
        // valore presente in una cella del voxel
        int cell_value;
        struct coords *old_position = (struct coords *)particles->list1[i];
        // controlla se la particella che si vuole muovere è stata cristallizzata
        getValue(space,*old_position, &cell_value);
        // particella  non cristallizzata
        if(cell_value != -1){
            // calcolo nuova posizione della particella
            struct coords new_position;
            move_particle_atomic(old_position, &new_position, rng, getSize(space));
            // controllo se la particella si sta spostando verso un cristallo
            getValue(space, new_position, &cell_value);
            //particella da cristallizzare
            if(cell_value ==-1){
                // aggiunge alla lista di particelle da cristallizzare
                dinamic_list_add(particles->freezed, old_position);
            }
            // si muove verso uno spazio vuoto
            else{
                // aggiorna la posizione della particella con la nuova posizione
                #pragma omp critical
                {
                    particles->last2++;
                    particles->list2[particles->last2] = particles->list1[i];
                }
                *particles->list1[i] = new_position;
            }
        }
        // particella cristallizzata
        else{
            free(particles->list1[i]);
        }
    }
    //      cristallizza le particelle che si trovano vicino a un cristallo
    // ciclo sulle particelle che devono essere cristallizzate
    for(int i = particles->freezed->last; i >= 0; i--) {
        // recupero l'elemento dalla lista dei freezed
        element e = dinamic_list_pop_last(particles->freezed);
        // controllo se ci sono stati errori
        if(e.error != 0) {
            continue;
        }
        // setto il valore della cella a -1
        setValue(space, *((struct coords *) e.value), -1);
        if(particles->last2 % ((int)(PART_NUM / 10)) == 0)
            printf("Crystalized: {%d, %d, %d} - %d / %d\n",((struct coords *)e.value)->x, ((struct coords *)e.value)->y, ((struct coords *)e.value)->z, particles->last2, PART_NUM);
        free(e.value);
    }
}
// void main(){
//     struct coords list1[PART_NUM];
//     int list1_last = -1;
//     struct coords list2[PART_NUM];
//     int list1_last = -1;
// }