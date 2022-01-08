#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>

#ifndef UTILS
    #include "../../../utils/utils.c"
#endif

#define MIN_BLOCK 4

////////////////////////
// pseudo random function
__device__ float random_float_cuda(int* rng) {
    int n  = (*rng << 13U) ^ *rng;
    (*rng) *= 6364136223846793005ULL;
    n = n * (n * n * 15731U + 789221U) + 1376312589U;
    return (float)((n & 0x7fffffffU)/((float)(0x7fffffff)));
}
/*
Prende un seed per il generatore random e la grandezza del voxel, scrive in una struttura particle 
la posizione della particella e l'rng. La particella viene posizionata in una delle facce del voxel.
*/
__device__ void get_random_position_cuda(struct coords voxel_size, int seed, struct particle *p) {
    p->rng = seed;
    p->coord.x = (int) (random_float_cuda(&p->rng) * voxel_size.x);
    p->coord.y = (int) (random_float_cuda(&p->rng) * voxel_size.y);
    p->coord.z = (int) (random_float_cuda(&p->rng) * voxel_size.z);
    switch((int) ((random_float_cuda(&(p->rng)) * 6))) {
        case 0:
            p->coord.x = 0;
            break;
        case 1:
            p->coord.x = voxel_size.x - 1;
            break;
        case 2:
            p->coord.y = 0;
            break;
        case 3:
            p->coord.y = voxel_size.y - 1;
            break;
        case 4:
            p->coord.z = 0;
            break;
        default: 
            p->coord.z = voxel_size.z - 1;
            break;
    }
     // tentativo di eliminare lo switch scartato perché dava risultati diversi

    /*    int data[] = {
        0, 1, 1,
        1, 0, 1,
        1, 1, 0,
        voxel_size.x - 1, 0, 0,
        0, voxel_size.y - 1, 0,
        0, 0, voxel_size.z - 1
    };
    int r = (int) (random_float_cuda(&p->rng) * 6);
    if(r < 3) {
        p->coord.x *= data[r * 3];
        p->coord.y *= data[r * 3 + 1];
        p->coord.z *= data[r * 3 + 2];
    }
    else {
        p->coord.x = data[r * 3] + data[r * 3 - 9] * p->coord.x;
        p->coord.y = data[r * 3 + 1] + data[r * 3 - 9 + 1] * p->coord.y;
        p->coord.z = data[r * 3 + 2] + data[r * 3 - 9 + 2] * p->coord.z;
    }*/


}
/*
Date le coordinate di una particella, un puntatore a un rng e la grandezza del voxel, mette nella coord out
le coordinate verso le quali si muoverà la particella.
Ci sono 27 possibili direzioni per la particella.
*/
__device__ void move_particle_cuda(struct coords c1, struct coords *out, int* rng, struct coords voxel_size){
    out->x = c1.x;
    out->y = c1.y;
    out->z = c1.z;
    int rx = ((int) (random_float_cuda(rng) * 3)) - 1;
    int ry = ((int) (random_float_cuda(rng) * 3)) - 1;
    int rz = ((int) (random_float_cuda(rng) * 3)) - 1;
    if((rx + out->x ) > 0 &&  (rx + out->x ) < voxel_size.x)
        out->x += rx;
    if((ry + out->y ) > 0 &&  (ry + out->y ) < voxel_size.y)
        out->y += ry;
    if((rz + out->z ) > 0 &&  (rz + out->z ) < voxel_size.z)
        out->z += rz;
    return;
}
/*
restituisce il thread ID
*/
__device__ int get_tid(){
    return ( gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x * blockDim.y * blockDim.z \
     + blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
}
/*
restituisce il numero di threads
*/
__device__ int get_num_threads(){
    return gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
}
/*
 muove le particelle di un passo.
*/
__global__ void step(int* voxel_d, struct coords voxel_size, struct particle** list_d, struct particle** freezed_d, int particle_number) {
    int tid = get_tid();
    int num_threads = get_num_threads();
    // ogni thread va a calcolare il movimento di alcune delle particelle.
    // thread contigui vanno a prendere particelle contigue
    // es: se ho tre thread e 10 particelle:
    // threads:     0  1  2 | 0  1  2 | 0  1  2 | 0
    // particelle:  0  1  2 | 3  4  5 | 6  7  8 | 9
    for(int i = tid; i < particle_number ; i += num_threads)  {
        struct particle *part = list_d[i];
        // prende il valore della cella 
        int cell_value = voxel_d[part->coord.x + part->coord.y * voxel_size.x + part->coord.z * voxel_size.x * voxel_size.y];
        
        if (cell_value != -1){  // la particella è in movimento
            struct coords new_pos;
            //prova a muovere la particella
            move_particle_cuda(part->coord, &new_pos, &part->rng, voxel_size);
            cell_value = voxel_d[new_pos.x + new_pos.y * voxel_size.x + new_pos.z * voxel_size.x * voxel_size.y];

            if((cell_value != -1)){ // la particella si è mossa in uno spazio vuoto
                // aggiorna la posizione
                part->coord = new_pos;
            }
            else { // la particella ha incontrato un cristallo
                // mette la particella nella lista di particelle da cristallizzare
                // e la rimuove dalla lista delle particelle che si stanno muovendo
                freezed_d[i] = part;
                list_d[i] = NULL;
            }
        }
        else{// la cella è già stata cristallizzata
                list_d[i] = NULL;
            }
    }
    return;
}

/*
 inizializza la lista delle particelle
*/
__global__ void init_particles_cuda(struct particle **list_d, int num, struct coords voxel_size , struct particle* particle_list) {
    // recupera il suo id
     int tid =   get_tid();
     int num_threads = get_num_threads();
    // i thread prendono alcune particelle e le inzializzano
    // Thread vicini prendono particelle vicine (guarda esempio di step())
    for(int i = tid; i < num; i += num_threads)  {
        list_d[i] = &particle_list[i];
        //      genera la particella a index i
        get_random_position_cuda(voxel_size, i + 1,list_d[i]);
    }
    
}
/*
 cristallizza le particelle
*/
__global__ void freeze(int* voxel_d, struct coords voxel_size, struct particle** list_d, struct particle** freezed_d, int particle_number_d){
    
    int tid =   get_tid();
    int num_threads = get_num_threads();
    // i thread prendono alcune particelle e le inzializzano
    // Thread vicini prendono particelle vicine (guarda esempio di step())
    for(int i = tid; i < particle_number_d; i += num_threads)  {
       if( freezed_d[i] != NULL ){// se la particella va cristallizzata
           //cristallizza e toglie il puntatore dalla lista delle particelle da cristallizzare
           voxel_d[freezed_d[i]->coord.x + freezed_d[i]->coord.y * voxel_size.x + freezed_d[i]->coord.z * voxel_size.x * voxel_size.y] = -1;
           
           freezed_d[i]=NULL;
       }
    }  
}
/*
Pippo_franco è la funzione utilizzata per far scorrere gli elementi di una lista verso sinisntra
in modo da non avere più buchi.
È considerabile come il passo "combina" di un algoritmo "divide et impera" bottom up.
Ciò che fa è, dato il livello a cui l'algoritmo deve combinare, calcola la grandezza 
dei sottoarray che devono essere rielaborati.
La particle list è la lista di puntatori a particelle su cui la funzione deve lavorare.
particle_number_d è il numero di particelle in particle_list_d.
v_array_d è un array che mantiene il numero di elementi che si trovano all'interno di un sottoarray.

leggenda: # -> elemento della lista piena
          - -> elemento della lista vuoto
array iniziale:                 # # - # - - # - # - - # - - - # # # - - # - # -

combina due array di 1 elementos
        level 0 out:                ## #- -- #- #- #- -- #- ## -- #- #-

combina due array di 2 elementi
        level 1 out:                   ###- #--- ##-- #--- ##-- ##--

combina due array di 4 elementi
level 2 out:                            ####---- ###----- ####----

combina due array di 8 elementi
level 3 out:                             #######--------- ####----

combina due array di 16 elementi
level 4 out:                              ###########-------------

i thread si dividono le celle durante il lavoro.
Se una cella si trova in un sottoarray multiplo di 2 ed è vuoto lavora, altrimenti no.

quando una casella è vuota e si trova in un sottoarray multiplo di 2 va a prendre un elemento pieno
dal sottoarray successivo (che di sicuro non lavora).

esempio con sottoarray di numghezza 16
threads 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15  
        #  #  #  #  -  -  -  -  -  -  -  -  -  -  -  - | #  #  #  #  #  #  #  #  -  -  -  -  -  -  -  -
                    v1                                   m                       v2
    in questo caso i thread da 0 a 3 non fanno niente. 
    I thread da 4 a 15 invece fanno un calcolo usando v1, v2 e m 
    per trovare la casella da cui spostare la particella

    v1 e v2 vengono recuperati da v_array_d.
    alla fine, v1 + v2 viene scritto in v_array_d in corrispondenza 
    dell'indice della prima particella del sottoarray di sinistra (durante i livelli
    pari viene messo ad offset 0 mentre durante i livelli dispari ad offset 1 in modo da 
    non sovrascriverli perchè alcuni dei thread potrebbero non aver finito di operare
    quando v1 + v2 viene scritto).
*/
__global__ void pippo_franco(struct particle** particle_list_d, int particle_number_d, int* v_array_d, int level) {
    int tid =   get_tid();
    
    int num_threads = get_num_threads();
    // caso base con un' unica particella che è stata eliminata
    if(particle_number_d == 1 && particle_list_d[0] == NULL) {
        v_array_d[0] = 0;
    }
    // primo passo di ridimensionamento 
    if(level == 0) {  
        // inizializza la lista v_array_d
        for(int i = tid * 2; i < (particle_number_d); i += (num_threads * 2)) {
            // controlla se la cella è vuota
            v_array_d[i] = (int) (particle_list_d[i] != NULL); 
            //controlla di non andare fuori dall'array
            if((i + 1) < particle_number_d) {
                // contiene il numero di celle piene per la coppia (i, i + 1)
                v_array_d[i] += (int) (particle_list_d[i + 1] != NULL);
                // se la cella i è vuota e quella i + 1 è piena fa uno swap
                if(particle_list_d[i] == NULL && v_array_d[i] == 1) {
                    // swap
                    particle_list_d[i] = particle_list_d[i + 1];
                    particle_list_d[i + 1] = NULL;
                }
               
            }
            
        }
        
    }
    // passi di ridimensionamento successivi
    else {
        // calcola lunghezza di blocco da ridimensionare
        int block_len = (1 << level);
        for(int i = tid; i < (particle_number_d); i += num_threads) {
           // controlla se il thread deve lavorare
            if(((int)(i / block_len)) % 2 == 0) {
                // trova il punto che divide 2 blocchi adiacenti di lunghezza block_len della lista da ordinare
                int m = block_len * ((int) (i / block_len) + 1);  
                // controlla di non andare fuori dall'array
                if(m < particle_number_d) {
                    // calcola il numero di particelle contenute in ciascuno dei 2 blocchi
                    int casella = m - block_len + ((int)(level % 2 == 0));
                    int v1 = v_array_d[casella];
                    int v2 = v_array_d[m + ((int)(level % 2 == 0))];
                    // se la cella controllata è vuota esegue uno swap con l'ultima cella piena del secondo blocco
                    if(particle_list_d[i] == NULL) {
                        // swap
                        int xx = m + v2 + (m - block_len) + v1 - i - 1;
                        
                        particle_list_d[i] = particle_list_d[xx];
                        particle_list_d[xx] = NULL;
                        
                    }
                    // un thread per blocco aggiorna il numero di particelle presenti nel nuovo blocco formato dall' unione dei 2 
                    if(i % block_len == 0) {
                        v_array_d[m - block_len + ((int)(level % 2 == 1))] = v1 + v2;
                    }
                    
                }
                // se il blocco non può essere unito con un altro aggiorna comunque il numero delle particelle in esso contenute
                else {
                    int xx = m - block_len + ((int)(level % 2 == 0));
                    int v1 = v_array_d[xx];
                    v_array_d[m - block_len + ((int)(level % 2 == 1))] = v1;
                    
                }
                
            }
            
        }
      
    }
}
// esegue una print dell' array di particelle
__global__ void print_the_fucking_array(struct particle** particle_list_d, int max) {
    if(get_tid() == 0)
        for(int i = 0; i < max; i++) 
            printf(" %d  -  %p\n", i, particle_list_d[i]);        
}

// esegue la simulazione di generazione cristallina e ritorna lo spazio con il cristallo generato sotto forma di lista di interi
int *parallel_dla_cuda(struct coords space_size, int chunk_size, int particle_number_h, unsigned int grid_w, unsigned int block_w) {
    // alloco lo spazio per il voxel
    struct coords voxel_size = {space_size.x * chunk_size, space_size.y * chunk_size, space_size.z * chunk_size};
    size_t space_number = voxel_size.x * voxel_size.y * voxel_size.z;

    int *space_h = (int *) calloc(space_number, sizeof(int));
    // piazza il primo cristallo
    space_h[voxel_size.x / 2 + (voxel_size.y / 2) * voxel_size.x + (voxel_size.z / 2) * voxel_size.x * voxel_size.y] = -1 ;
    // alloca la memoria in  cuda e copia i dati che serviranno ai vari kernel
    cudaError_t  err;
    // alloca lo spazio di simulazione
    int *voxel_d;
    
    err = cudaMalloc((void **) &voxel_d, space_number * sizeof(int));
    if(err < 0) {
        fprintf(stderr, "Malloc Failed");
        exit(1);
    }
    
    err = cudaMemcpy(voxel_d, space_h, space_number * sizeof(int), cudaMemcpyHostToDevice);

    if(err < 0) {
        fprintf(stderr, "Malloc Failed");
        exit(1);
    }

    // Alloco lo spazio per le particelle
    struct particle **particles_d;
    err = cudaMalloc(&particles_d, sizeof(struct particle *) * particle_number_h);

    struct particle *particle_list_d;
    err = cudaMalloc(&particle_list_d, sizeof(struct particle) * particle_number_h);

    if(err < 0) {
        fprintf(stderr, "Malloc Failed");
        exit(1);
    }
    // imposta la grandezza della griglia e dei blocchi
    dim3 GridDim = {grid_w, grid_w, grid_w};
    dim3 BlockDim = {block_w, block_w, block_w};
    // inizializza le particelle
    init_particles_cuda<<<GridDim, BlockDim>>>(particles_d, particle_number_h, voxel_size, particle_list_d);
    // alloca la lista di particelle da cristallizzare
    struct particle **freezed_d;
    err = cudaMalloc(&freezed_d, sizeof(struct particle *) * particle_number_h);

    if(err < 0) {
        fprintf(stderr, "Malloc Failed");
        exit(1);
    }

    // Alloco lo spazio per i v
    int *v_array_d;
    err = cudaMalloc(&v_array_d, sizeof(int) * particle_number_h);

    if(err < 0) {
        fprintf(stderr, "Malloc Failed");
        exit(1);
    }
    
    // esegue la simulazione finché non si sono cristallizzate tutte le particelle
    while(particle_number_h > 0) {
        // muove le particelle
        step<<<GridDim, BlockDim>>>(voxel_d, voxel_size, particles_d, freezed_d, particle_number_h);
        cudaDeviceSynchronize();
        // cristallizza le particelle
        freeze<<<GridDim, BlockDim>>>(voxel_d, voxel_size, particles_d, freezed_d, particle_number_h);
        cudaDeviceSynchronize();
        // ridimensiona la lista di particelle eliminando quelle cristallizzate
        int logarithm =  (int) ceil(log2(particle_number_h)) - 1 + ((int)particle_number_h == 1);
        for(int level = 0; level <= logarithm; level++) {
            pippo_franco<<<GridDim, BlockDim>>>(particles_d, particle_number_h, v_array_d, level);
        }
        // copia il numero delle particelle rimaste dal device.
        cudaMemcpy(&particle_number_h, v_array_d + ((int)(logarithm % 2 == 1)), sizeof(int), cudaMemcpyDeviceToHost);
        // print per sapere il numero di particelle rimaste
        /* if(particle_number_h % 1000 == 1) printf("Particle_number: %d\n", particle_number_h);*/
        // ridimensiono il numero di threads
        if (GridDim.x != MIN_BLOCK && (GridDim.x * GridDim.x * GridDim.x * block_w *block_w * block_w) > particle_number_h) {
            GridDim.x = max(MIN_BLOCK,(int) (ceil(GridDim.x / 2)));
            GridDim.y = max(MIN_BLOCK,(int) (ceil(GridDim.y / 2)));
            GridDim.z = max(MIN_BLOCK,(int) (ceil(GridDim.z / 2)));
        }
    }
    
    cudaMemcpy(space_h,  voxel_d, space_number *sizeof(int), cudaMemcpyDeviceToHost);
    // libera la memoria della gpu
    cudaFree(freezed_d);
    cudaFree(voxel_d);
    cudaFree(v_array_d);
    cudaFree(particles_d);
    cudaFree(particle_list_d);
    
    return space_h;
}


