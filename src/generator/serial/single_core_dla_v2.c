#include <stdio.h>

#ifndef UTILS
    #include "../../../utils/utils.c"
#endif
#ifndef DINAMIC_LISTS
    #include "../../../utils/dinamic_list.h"
#endif

/*
Prende un seed per il generatore random e la grandezza del voxel, ritornando una struttura particle al cui interno ci sarà
la posizione della particella e l'rng. La particella viene posizionata in una delle facce del voxel.
*/
struct particle *get_random_position(struct coords voxel_size, int seed) {
    struct particle *ret = (struct particle *) malloc(sizeof(struct particle));
    ret->rng = seed;
    ret->coord.x = (int) ((random_float(&ret->rng) * voxel_size.x));
    ret->coord.y = (int) ((random_float(&ret->rng) * voxel_size.y));
    ret->coord.z = (int) ((random_float(&ret->rng) * voxel_size.z));
    switch((int) ((random_float(&ret->rng) * 6))) {
        case 0:
            ret->coord.x = 0;
            break;
        case 1:
            ret->coord.x = voxel_size.x - 1;
            break;
        case 2:
            ret->coord.y =0;
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

/*
Date le coordinate di una pparticella, un puntatore a un rng e la grandezza del voxel, mette nella coord c1
le coordinate verso le quali si muoverà la particella.
Ci sono 27 possibili direzioni per la particella.
*/
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

/*
Inizializza num particelle nel voxel v e le inserisce in list.
*/
void init_particles(struct particle **list, int num, struct voxel *v) {
    struct coords size = getSize(v);
    for (int i=0; i<num; i++) {
        list[i] = get_random_position(size, i + 1);
    }
}

/*
Muove le particelle di un passo.
Dato un voxel e una lista di particelle, fa avanzare le particelle di uno step e, se stanno per spostarsi su un cristallo, le cristallizza a loro volta
*/
void single_core_dla(struct voxel *space, struct particle_lists *particles) {
    struct coords voxel_size = getSize(space);
    // muove le particelle
    // per ogni particella
    for (int i = 0; i <= particles->last1; i++){
        // valore presente in una cella del voxel
        int cell_value;
        struct particle *part = (struct particle *)particles->list1[i];
        // controlla se la particella che si vuole muovere si trova in una casella che è stata cristallizzata
        getValue(space, part->coord, &cell_value);

        if(cell_value != -1){   // casella  non cristallizzata
            // calcolo nuova posizione della particella
            struct coords new_position;
            move_particle(&(part->coord), &new_position, &(part->rng), voxel_size);
            // controllo se la particella si sta spostando verso un cristallo
            getValue(space, new_position, &cell_value);
            
            if(cell_value ==-1){  // la particella è da cristallizzare
                // aggiunge alla lista di particelle da cristallizzare
                dinamic_list_add(particles->freezed, part);
                // elimina la particella dalla lista di particelle attive
                particles->list1[i] = NULL;
            }
            else{// si muove verso uno spazio vuoto
                // aggiorna la posizione della particella con la nuova posizione
                part->coord = new_position;
            }
        }        
        else{  // casella cristallizzata
            // elimina la particella
            free(particles->list1[i]);
            particles->list1[i] = NULL;
        }
    }


    // cristallizza le particelle che si trovano nella lista freezed
    for(int i = particles->freezed->last; i >= 0; i--) {
        // recupero l'elemento dalla lista dei freezed
        element e = dinamic_list_pop_last(particles->freezed);
        // controllo se ci sono stati errori
        if(e.error != 0) {
            continue;
        }
        // setto il valore della cella a -1
        setValue(space, ((struct particle *) e.value)->coord, -1);

        /*
        print per sapere a che punto sta l'algoritmo

        if(particles->last1 % ((int)(PART_NUM / 10)) == 0)
            printf("Crystalized: {%d, %d, %d} - %d / %d\n",((struct particle *)e.value)->coord.x, ((struct particle *)e.value)->coord.y, ((struct particle *)e.value)->coord.z, particles->last1, PART_NUM);
        */

        // libero la memoria
        free(e.value);
    }
    // elimina i buchi lasciati dalle particelle che sono state eliminate o che si sono cristallizzate
    int last = -1;
    for(int i = 0; i <= particles->last1; i++){
        if (particles->list1[i] != NULL)
            particles->list1[++last] = particles->list1[i];
    }
    particles->last1 = last;
}

