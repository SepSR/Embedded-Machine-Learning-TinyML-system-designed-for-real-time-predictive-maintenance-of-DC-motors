#pragma once
#include <cstdarg>
namespace Eloquent {
    namespace ML {
        namespace Port {
            class VibrationClassifier {
                public:
                    /**
                    * Predict class for features vector
                    */
                    int predict(float *x) {
                        float kernels[85] = { 0 };
                        float decisions[1] = { 0 };
                        int votes[2] = { 0 };
                        kernels[0] = compute_kernel(x,   3322.0  , 0.6644  , 1719.0 );
                        kernels[1] = compute_kernel(x,   3354.0  , 0.6708  , 1703.0 );
                        kernels[2] = compute_kernel(x,   3456.0  , 0.6912  , 2925.0 );
                        kernels[3] = compute_kernel(x,   3319.0  , 0.6638  , 2330.0 );
                        kernels[4] = compute_kernel(x,   3632.0  , 0.7264  , 2486.0 );
                        kernels[5] = compute_kernel(x,   3538.0  , 0.7076  , 2034.0 );
                        kernels[6] = compute_kernel(x,   3393.0  , 0.6786  , 1856.0 );
                        kernels[7] = compute_kernel(x,   3377.0  , 0.6754  , 1863.0 );
                        kernels[8] = compute_kernel(x,   3373.0  , 0.6746  , 1817.0 );
                        kernels[9] = compute_kernel(x,   3328.0  , 0.6656  , 1993.0 );
                        kernels[10] = compute_kernel(x,   3307.0  , 0.6614  , 1677.0 );
                        kernels[11] = compute_kernel(x,   3494.0  , 0.6988  , 1914.0 );
                        kernels[12] = compute_kernel(x,   3391.0  , 0.6782  , 1800.0 );
                        kernels[13] = compute_kernel(x,   3503.0  , 0.7006  , 1638.0 );
                        kernels[14] = compute_kernel(x,   3344.0  , 0.6688  , 1781.0 );
                        kernels[15] = compute_kernel(x,   3367.0  , 0.6734  , 1809.0 );
                        kernels[16] = compute_kernel(x,   3536.0  , 0.7072  , 1509.0 );
                        kernels[17] = compute_kernel(x,   3512.0  , 0.7024  , 1736.0 );
                        kernels[18] = compute_kernel(x,   3340.0  , 0.668  , 2081.0 );
                        kernels[19] = compute_kernel(x,   3434.0  , 0.6868  , 1698.0 );
                        kernels[20] = compute_kernel(x,   3490.0  , 0.698  , 1673.0 );
                        kernels[21] = compute_kernel(x,   3569.0  , 0.7138  , 1583.0 );
                        kernels[22] = compute_kernel(x,   3543.0  , 0.7086  , 1957.0 );
                        kernels[23] = compute_kernel(x,   3350.0  , 0.67  , 1885.0 );
                        kernels[24] = compute_kernel(x,   3376.0  , 0.6752  , 1440.0 );
                        kernels[25] = compute_kernel(x,   3525.0  , 0.705  , 1100.0 );
                        kernels[26] = compute_kernel(x,   3492.0  , 0.6984  , 1964.0 );
                        kernels[27] = compute_kernel(x,   3321.0  , 0.6642  , 1824.0 );
                        kernels[28] = compute_kernel(x,   3423.0  , 0.6846  , 1217.0 );
                        kernels[29] = compute_kernel(x,   3511.0  , 0.7022  , 1485.0 );
                        kernels[30] = compute_kernel(x,   3504.0  , 0.7008  , 1495.0 );
                        kernels[31] = compute_kernel(x,   3558.0  , 0.7116  , 1484.0 );
                        kernels[32] = compute_kernel(x,   3306.0  , 0.6612  , 2016.0 );
                        kernels[33] = compute_kernel(x,   3488.0  , 0.6976  , 1618.0 );
                        kernels[34] = compute_kernel(x,   3482.0  , 0.6964  , 1679.0 );
                        kernels[35] = compute_kernel(x,   3452.0  , 0.6904  , 1430.0 );
                        kernels[36] = compute_kernel(x,   3525.0  , 0.705  , 2110.0 );
                        kernels[37] = compute_kernel(x,   3369.0  , 0.6738  , 2308.0 );
                        kernels[38] = compute_kernel(x,   3394.0  , 0.6788  , 1538.0 );
                        kernels[39] = compute_kernel(x,   3413.0  , 0.6826  , 1424.0 );
                        kernels[40] = compute_kernel(x,   3433.0  , 0.6866  , 1552.0 );
                        kernels[41] = compute_kernel(x,   3555.0  , 0.711  , 1536.0 );
                        kernels[42] = compute_kernel(x,   3440.0  , 0.688  , 1634.0 );
                        kernels[43] = compute_kernel(x,   3533.0  , 0.7066  , 1294.0 );
                        kernels[44] = compute_kernel(x,   3332.0  , 0.6664  , 1578.0 );
                        kernels[45] = compute_kernel(x,   3497.0  , 0.6994  , 1632.0 );
                        kernels[46] = compute_kernel(x,   3433.0  , 0.6866  , 2323.0 );
                        kernels[47] = compute_kernel(x,   3392.0  , 0.6784  , 1621.0 );
                        kernels[48] = compute_kernel(x,   3324.0  , 0.6648  , 2280.0 );
                        kernels[49] = compute_kernel(x,   3569.0  , 0.7138  , 1889.0 );
                        kernels[50] = compute_kernel(x,   3414.0  , 0.6828  , 1344.0 );
                        kernels[51] = compute_kernel(x,   3479.0  , 0.6958  , 1456.0 );
                        kernels[52] = compute_kernel(x,   3529.0  , 0.7058  , 2119.0 );
                        kernels[53] = compute_kernel(x,   3365.0  , 0.673  , 1659.0 );
                        kernels[54] = compute_kernel(x,   4112.0  , 0.8224  , 1077.0 );
                        kernels[55] = compute_kernel(x,   3826.0  , 0.7652  , 1405.0 );
                        kernels[56] = compute_kernel(x,   3975.0  , 0.795  , 1337.0 );
                        kernels[57] = compute_kernel(x,   3657.0  , 0.7314  , 1051.0 );
                        kernels[58] = compute_kernel(x,   4196.0  , 0.8392  , 1338.0 );
                        kernels[59] = compute_kernel(x,   3006.0  , 0.6012  , 2050.0 );
                        kernels[60] = compute_kernel(x,   3266.0  , 0.6532  , 2679.0 );
                        kernels[61] = compute_kernel(x,   2838.0  , 0.5676  , 1959.0 );
                        kernels[62] = compute_kernel(x,   3860.0  , 0.772  , 1352.0 );
                        kernels[63] = compute_kernel(x,   3100.0  , 0.62  , 1252.0 );
                        kernels[64] = compute_kernel(x,   2462.0  , 0.4924  , 1519.0 );
                        kernels[65] = compute_kernel(x,   4166.0  , 0.8332  , 1018.0 );
                        kernels[66] = compute_kernel(x,   3798.0  , 0.7596  , 1001.0 );
                        kernels[67] = compute_kernel(x,   3801.0  , 0.7602  , 1084.0 );
                        kernels[68] = compute_kernel(x,   1917.0  , 0.3834  , 7183.0 );
                        kernels[69] = compute_kernel(x,   3721.0  , 0.7442  , 1057.0 );
                        kernels[70] = compute_kernel(x,   2966.0  , 0.5932  , 1997.0 );
                        kernels[71] = compute_kernel(x,   4180.0  , 0.836  , 980.0 );
                        kernels[72] = compute_kernel(x,   2306.0  , 0.4612  , 4267.0 );
                        kernels[73] = compute_kernel(x,   3991.0  , 0.7982  , 1747.0 );
                        kernels[74] = compute_kernel(x,   2304.0  , 0.4608  , 3022.0 );
                        kernels[75] = compute_kernel(x,   4188.0  , 0.8376  , 984.0 );
                        kernels[76] = compute_kernel(x,   2331.0  , 0.4662  , 2890.0 );
                        kernels[77] = compute_kernel(x,   3073.0  , 0.6146  , 2019.0 );
                        kernels[78] = compute_kernel(x,   3940.0  , 0.788  , 1033.0 );
                        kernels[79] = compute_kernel(x,   3909.0  , 0.7818  , 1028.0 );
                        kernels[80] = compute_kernel(x,   3923.0  , 0.7846  , 1041.0 );
                        kernels[81] = compute_kernel(x,   2584.0  , 0.5168  , 1344.0 );
                        kernels[82] = compute_kernel(x,   3808.0  , 0.7616  , 1416.0 );
                        kernels[83] = compute_kernel(x,   3868.0  , 0.7736  , 987.0 );
                        kernels[84] = compute_kernel(x,   3791.0  , 0.7582  , 1055.0 );
                        float decision = -1946.846523584804;
                        decision = decision - ( + kernels[0] * -7.928571428571  + kernels[1] * -7.928571428571  + kernels[2] * -7.928571428571  + kernels[3] * -7.928571428571  + kernels[4] * -0.06360984429  + kernels[5] * -7.928571428571  + kernels[6] * -7.928571428571  + kernels[7] * -7.928571428571  + kernels[8] * -7.928571428571  + kernels[9] * -7.928571428571  + kernels[10] * -5.225683935827  + kernels[11] * -7.928571428571  + kernels[12] * -7.928571428571  + kernels[13] * -7.928571428571  + kernels[14] * -7.928571428571  + kernels[15] * -7.928571428571  + kernels[16] * -7.928571428571  + kernels[17] * -7.928571428571  + kernels[18] * -7.928571428571  + kernels[19] * -7.928571428571  + kernels[20] * -7.928571428571  + kernels[21] * -7.928571428571  + kernels[22] * -7.928571428571  + kernels[23] * -7.928571428571  + kernels[24] * -7.928571428571  + kernels[25] * -7.928571428571  + kernels[26] * -7.928571428571  + kernels[27] * -7.928571428571  + kernels[28] * -7.928571428571  + kernels[29] * -7.928571428571  + kernels[30] * -7.928571428571  + kernels[31] * -7.928571428571  + kernels[32] * -7.928571428571  + kernels[33] * -7.928571428571  + kernels[34] * -7.928571428571  + kernels[35] * -7.928571428571  + kernels[36] * -7.928571428571  + kernels[37] * -7.928571428571  + kernels[38] * -7.928571428571  + kernels[39] * -7.928571428571  + kernels[40] * -7.928571428571  + kernels[41] * -7.928571428571  + kernels[42] * -7.928571428571  + kernels[43] * -7.928571428571  + kernels[44] * -7.928571428571  + kernels[45] * -7.928571428571  + kernels[46] * -7.928571428571  + kernels[47] * -7.928571428571  + kernels[48] * -7.928571428571  + kernels[49] * -3.079772307363  + kernels[50] * -7.928571428571  + kernels[51] * -7.928571428571  + kernels[52] * -7.928571428571  + kernels[53] * -7.928571428571 );
                        decision = decision - ( + kernels[54] * 13.536585365854  + kernels[55] * 13.536585365854  + kernels[56] * 13.536585365854  + kernels[57] * 13.536585365854  + kernels[58] * 6.628647969014  + kernels[59] * 13.536585365854  + kernels[60] * 13.536585365854  + kernels[61] * 13.536585365854  + kernels[62] * 13.536585365854  + kernels[63] * 13.536585365854  + kernels[64] * 13.536585365854  + kernels[65] * 13.536585365854  + kernels[66] * 13.536585365854  + kernels[67] * 13.536585365854  + kernels[68] * 13.536585365854  + kernels[69] * 13.536585365854  + kernels[70] * 13.536585365854  + kernels[71] * 13.536585365854  + kernels[72] * 13.536585365854  + kernels[73] * 13.536585365854  + kernels[74] * 13.536585365854  + kernels[75] * 13.536585365854  + kernels[76] * 13.536585365854  + kernels[77] * 13.536585365854  + kernels[78] * 13.536585365854  + kernels[79] * 13.536585365854  + kernels[80] * 13.536585365854  + kernels[81] * 13.536585365854  + kernels[82] * 13.536585365854  + kernels[83] * 13.536585365854  + kernels[84] * 13.536585365854 );

                        return decision > 0 ? 0 : 1;
                    }

                protected:
                    /**
                    * Compute kernel between feature vector and support vector.
                    * Kernel type: linear
                    */
                    float compute_kernel(float *x, ...) {
                        va_list w;
                        va_start(w, x);
                        float kernel = 0.0;

                        for (uint16_t i = 0; i < 3; i++) {
                            kernel += x[i] * va_arg(w, double);
                        }

                        return kernel;
                    }
                };
            }
        }
    }