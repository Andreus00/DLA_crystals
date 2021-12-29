#include "../utils/utils.c"
#include "../utils/dinamic_list.h"
int main(int argc, char const *argv[])
{
    int first;
    int rng = 3;
    first = random_float(&rng);
    int c = 0;
    while(random_float(&rng)) { printf("%f\n", random_float(&rng));};
    return 0;
}
