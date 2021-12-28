#include "../utils/utils.c"

#define VOXEL_SIZE_STRUCT {1,1,1}


int main(int argc, char const *argv[])
{
    // initialization test
    struct voxel v;
    struct coords c = VOXEL_SIZE_STRUCT;
    printf("INITIALIZATION OF THE VOXEL. SIZE: %d %d %d\n", c.x, c.y, c.z);
    init_voxel(&v, c);

    // setValue() test
    struct coords cc = {10,0,4};
    if(setValue(&v, cc, 1) < 0) {
        printf("ERROR\n");
    };
    printf("SETTING THE VALUE IN THE CELL {%d, %d, %d}\n", cc.x, cc.y, cc.z);

    // getValue() test
    int out;
    int err = getValue(&v, cc, &out);
    printf("GETTING THE VALUE IN THE CELL {%d, %d, %d}\nTHE VALUE IS %d. ERROR FLAG: %d\n", cc.x, cc.y, cc.z, out, err);

    // set fail test
    struct coords cc2 = {16,0,0};
    if(setValue(&v, cc2, 1) < 0) {
        printf("FAIL TEST: PASSED\n");
    };

    return 0;
}
