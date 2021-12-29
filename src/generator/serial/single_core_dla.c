#include <stdio.h>

#include "../../../utils/utils.c"
#include "../../../utils/dinamic_list.h"
#define PART_NUM 100

struct coords *get_random_position(struct coords voxel_size, int* rng) {
    struct coords *ret = (struct coords *) malloc(sizeof(struct coords));
    ret->x = (int) (random_float(rng) * voxel_size.x);
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
struct coords move_particle(struct coords *c1,int* rng, struct coords voxel_size){

    int dir = (int) random_float(rng)*6;
    struct coords c = {c1->x, c1->y, c1->z};
    switch (dir){
        case 0:
            if(c.x < voxel_size.x - 1)
                c.x++;
            break;
        case 1:
            if(c.x > 0)
                c.x--;
            break;
        case 2:
            if(c.y < voxel_size.y - 1)
                c.y++;
            break;
        case 3:
            if(c.y > 0)
                c.y--;
            break;
        case 4:
            if(c.z < voxel_size.z - 1)
                c.z++;
            break;
        default:
            if(c.z > 0)
                c.z--;
            break;
    };
    return c;
}

void init_particles(dinamic_list *list, int num, int* rng, struct voxel *v) {
    for (int i=0; i<num; i++) {
        dinamic_list_add(list, get_random_position(getSize(v),rng));
    }
}

int main(int argc, char const *argv[])
{
    // inizializzo l'rng
    
    int rng = 69420;
    
    // inizializzazione del volxel

    struct voxel space;
    struct coords space_size = {10,10,10};
    init_voxel(&space, space_size);


    // inizializzazione del cristallo iniziale

    struct coords initial_crystal = {5*16, 5*16, 5*16};
    setValue(&space, initial_crystal, -1);

    // inizializzazione delle particelle

    dinamic_list *particle_list = dinamic_list_new();

    // inizializzazione della lista delle particelle da cristallizzare

    dinamic_list *freezed = dinamic_list_new();
    
    init_particles(particle_list, PART_NUM, &rng, &space);

    while(particle_list->last != 0) {
        //      muove le particelle
        // per ogni particella
        for (int i = particle_list->last; i >= 0; i--){
            // valore presente in una cella del voxel
            int cell_value;
            // controlla se la particella che si vuole muovere è stata cristallizzata
            getValue(&space,*((struct coords *)particle_list->list[i]), &cell_value);
            // particella  non cristallizzata
            if(cell_value != -1){
                // calcolo nuova posizione della particella
                struct coords new_position = move_particle(particle_list->list[i], &rng, getSize(&space));
                // controllo se la particella si sta spostando verso un cristallo
                getValue(&space, new_position, &cell_value);
                //particella da cristallizzare
                if(cell_value ==-1){
                    // aggiunge alla lista di particelle da cristallizzare
                    dinamic_list_add(freezed, particle_list->list[i]);
                    // elimina la particella dalla lista di particelle attive
                    dinamic_list_fast_pop(particle_list, i);
                    
                }
                // si muove verso uno spazio vuoto
                else{
                    // aggiorna la posizione della particella con la nuova posizione
                    *((struct coords *)particle_list->list[i]) = new_position;
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
            setValue(&space, *((struct coords *) e.value), -1);
            free(e.value);
        }
    }
    return 0;
}
