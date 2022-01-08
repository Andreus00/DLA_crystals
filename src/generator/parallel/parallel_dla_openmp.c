
#ifndef UTILS
    #include "../../../utils/utils.c"
#endif
#ifndef DINAMIC_LISTS
    #include "../../../utils/dinamic_list.h"
#endif
#include <omp.h>

/*
restituisce una particella posizionata randomicamente su una delle facce del cubo rappresentante lo spazio su cui avviene la simulazione
*/
struct particle *get_random_position_omp(struct coords voxel_size, int seed) {
    //alloca la particella
    struct particle *ret = (struct particle *) malloc(sizeof(struct particle));
    //setta il random
    ret->rng = seed;
    //genera coordinate casuali
    ret->coord.x = (int) (random_float(&ret->rng) * voxel_size.x);
    ret->coord.y = (int) (random_float(&ret->rng) * voxel_size.y);
    ret->coord.z = (int) (random_float(&ret->rng) * voxel_size.z);
    //posiziona la particella su una faccia del cubo di simulazione 
    switch((int) ((random_float(&ret->rng) * 6))) {
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
/*
sposta una particella in una delle 27 possibili direzioni
*/
void move_particle_omp(struct coords *c1, struct coords *out, int* rng, struct coords voxel_size){
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
// inizializza la lista di particelle
void init_particles_parallel(struct particle **list, int num, struct voxel *v, const int num_threads) {
    struct coords size = getSize(v);
    // i threads inizializzano in parallelo le particelle
    #pragma omp parallel for  num_threads(num_threads)
    for (int i=0; i<num; i++) {
        list[i] = get_random_position_omp(size, i + 1);
    } 
}
/*
Esegue un passo di computazione seriale.
Dato un voxel e una lista di particelle, fa avanzare le particelle di uno step e, se stanno per spostarsi su un cristallo, le cristallizza a loro volta
*/
void parallel_dla_openmp(struct voxel *space, struct particle_lists *particles, const int num_threads) { 
    omp_set_num_threads(num_threads);
    
    struct coords voxel_size = getSize(space);
    
    
    // inizia la parte parallela
    #pragma omp parallel
    {
        //per ogni particella il thread che la prende la fa muovere
        #pragma omp for schedule(dynamic, 100)
            for (int i = 0; i <= particles->last1; i++) {
                // valore presente in una cella del voxel
                int cell_value;
                //particella
                struct particle *part = (struct particle *)particles->list1[i];
                // controlla se la particella che si vuole muovere è stata cristallizzata
                getValue(space, part->coord, &cell_value);
                // particella  non cristallizzata
                if(cell_value != -1) {
                    // calcolo nuova posizione della particella
                    struct coords new_position;
                    move_particle_omp(&(part->coord), &new_position, &(part->rng), voxel_size);
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
        
        
        // aspetta che tutte le particelle si siano mosse
        #pragma omp barrier

        //      cristallizza le particelle che si trovano vicino a un cristallo
        // ciclo sulle particelle che devono essere cristallizzate
        #pragma omp for
        for(int i = 0; i  <= particles->freezed->last; i++) {
            
            // setto il valore della cella a -1
            setValue(space, ((struct particle *) particles->freezed->list[i])->coord, -1);
            /*
            print per controllare a che punto si trova l'esecuzione

            if(particles->last1 % ((int)(PART_NUM / 10)) == 0)
                printf("%d / %d\n", particles->last1, PART_NUM);
            */
            free(particles->freezed->list[i]);
        }
        
        // a questo punto un core va a eliminare i buchi che si sono formati nella lista delle particelle
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
    // fa il clear della lista dei freezed
    dinamic_list_clear(particles->freezed);
}
