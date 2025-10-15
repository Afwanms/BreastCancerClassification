#pragma once
namespace Eloquent {
    namespace ML {
        namespace Port {
            class DecisionTree {
                public:
                    /**
                    * Predict class for features vector
                    */
                    int predict(float *x) {
                        if (x[0] <= -1.4275631308555603) {
                            return 1;
                        }

                        else {
                            if (x[1] <= 1.60970938205719) {
                                if (x[1] <= 0.17963892221450806) {
                                    if (x[2] <= -0.3461698591709137) {
                                        if (x[2] <= -0.4527123421430588) {
                                            if (x[1] <= -0.7079291343688965) {
                                                if (x[0] <= -1.3317546248435974) {
                                                    return 0;
                                                }

                                                else {
                                                    if (x[0] <= 0.8220424056053162) {
                                                        if (x[2] <= -0.8228738009929657) {
                                                            if (x[1] <= -1.249755620956421) {
                                                                return 1;
                                                            }

                                                            else {
                                                                return 0;
                                                            }
                                                        }

                                                        else {
                                                            return 1;
                                                        }
                                                    }

                                                    else {
                                                        return 0;
                                                    }
                                                }
                                            }

                                            else {
                                                return 0;
                                            }
                                        }

                                        else {
                                            if (x[0] <= 0.017722252756357193) {
                                                if (x[0] <= -0.32317138463258743) {
                                                    return 1;
                                                }

                                                else {
                                                    return 0;
                                                }
                                            }

                                            else {
                                                return 1;
                                            }
                                        }
                                    }

                                    else {
                                        return 0;
                                    }
                                }

                                else {
                                    if (x[0] <= -0.3231169283390045) {
                                        if (x[1] <= 0.9138228893280029) {
                                            return 0;
                                        }

                                        else {
                                            return 1;
                                        }
                                    }

                                    else {
                                        return 1;
                                    }
                                }
                            }

                            else {
                                return 0;
                            }
                        }
                    }

                protected:
                };
            }
        }
    }