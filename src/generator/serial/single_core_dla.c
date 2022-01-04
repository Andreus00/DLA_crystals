#include <stdio.h>

#ifndef UTILS
    #include "../../../utils/utils.c"
#endif
#ifndef DINAMIC_LISTS
    #include "../../../utils/dinamic_list.h"
#endif

#define PART_NUM 8000

struct coords *get_random_position(struct coords voxel_size, int* rng) {
    struct coords *ret = (struct coords *) malloc(sizeof(struct coords));
    ret->x = (int) (atomic_random_float(rng) * voxel_size.x);
    ret->y = (int) (random_float(rng) * voxel_size.y);
    ret->z = (int) (random_float(rng) * voxel_size.z);
    switch((int) random_float(rng) * 6) {
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
void move_particle(struct coords *c1, struct coords *out, int* rng, struct coords voxel_size){

    out->x = c1->x;
    out->y = c1->y;
    out->z = c1->z;
    int rx = ((int) (random_float(rng) * 3)) - 1;
    int ry = ((int) (random_float(rng) * 3)) - 1;
    int rz = ((int) (random_float(rng) * 3)) - 1;
    if((rx + out->x ) > 0 &&  (rx + out->x ) < voxel_size.x)
        out->x += rx;
    if((ry + out->y ) > 0 &&  (ry + out->y ) < voxel_size.y)
        out->y += ry;
    if((rz + out->z ) > 0 &&  (rz + out->z ) < voxel_size.z)
        out->z += rz;
    return;
}

void init_particles(dinamic_list *list, int num, int* rng, struct voxel *v) {
    for (int i=0; i<num; i++) {
        dinamic_list_add(list, get_random_position(getSize(v),rng));
    }
}

void single_core_dla(struct voxel *space, dinamic_list *particle_list, dinamic_list *freezed, int *rng) {
    // muove le particelle
    // per ogni particella
    for (int i = particle_list->last; i >= 0; i--){
        // valore presente in una cella del voxel
        int cell_value;
        struct coords *old_position = (struct coords *)particle_list->list[i];
        // controlla se la particella che si vuole muovere è stata cristallizzata
        getValue(space,*old_position, &cell_value);
        // particella  non cristallizzata
        if(cell_value != -1){
            // calcolo nuova posizione della particella
            struct coords new_position;
            move_particle(old_position, &new_position, rng, getSize(space));
            // controllo se la particella si sta spostando verso un cristallo
            getValue(space, new_position, &cell_value);
            //particella da cristallizzare
            if(cell_value ==-1){
                // aggiunge alla lista di particelle da cristallizzare
                dinamic_list_add(freezed, old_position);
                // elimina la particella dalla lista di particelle attive
                dinamic_list_fast_pop(particle_list, i);
            }
            // si muove verso uno spazio vuoto
            else{
                // aggiorna la posizione della particella con la nuova posizione
                *old_position = new_position;
            }
        }
        // particella cristallizzata
        else{
            element e = dinamic_list_fast_pop(particle_list, i);
            free(e.value);
        }
    }
    //      cristallizza le particelle che si trovano vicino a un cristallo
    // ciclo sulle particelle che devono essere cristallizzate
    for(int i = freezed->last; i >= 0; i--) {
        // recupero l'elemento dalla lista dei freezed
        element e = dinamic_list_pop_last(freezed);
        // controllo se ci sono stati errori
        if(e.error != 0) {
            continue;
        }
        // setto il valore della cella a -1
        setValue(space, *((struct coords *) e.value), -1);
        if(particle_list->last % ((int)(PART_NUM / 10)) == 0)
            printf("Crystalized: {%d, %d, %d} - %ld / %d\n",((struct coords *)e.value)->x, ((struct coords *)e.value)->y, ((struct coords *)e.value)->z, particle_list->last, PART_NUM);
        free(e.value);
    }
}


//commentato per problema doppio main

// int main(int argc, char const *argv[])
// {
//     int rng = 69420;
    
//     // inizializzazione del volxel

//     struct voxel space;
//     struct coords space_size = {10,10,10};
//     init_voxel(&space, space_size);


//     // inizializzazione del cristallo iniziale

//     struct coords initial_crystal = {5 * 16, 5 * 16, 5 * 16};
//     setValue(&space, initial_crystal, -1);

//     // inizializzazione delle particelle

//     dinamic_list *particle_list = dinamic_list_new();

//     // inizializzazione della lista delle particelle da cristallizzare

//     dinamic_list *freezed = dinamic_list_new();
    
//     init_particles(particle_list, PART_NUM, &rng, &space);

//     while(particle_list->last >= 0) {
//         single_core_dla(&space, particle_list, freezed, &rng);
//     }
    
    
//     return 0;
// }

