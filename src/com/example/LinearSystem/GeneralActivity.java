package com.example.LinearSystem;

import android.app.Activity;
import android.os.Bundle;

import java.util.Arrays;
import java.util.Random;

public class GeneralActivity extends Activity {

    private static final Random random = new Random();

    private static class InconsistentInputException extends Exception {
        public InconsistentInputException(String s) {
            super(s);
        }
    }

    private static class Matrix implements Cloneable {
        private int n;
        private double[][] a;
        private double[] multiplier;

        private double norm = -1;

        public Matrix(int size) {
            n = size;
            a = new double[n][n];
            multiplier = new double[n];
            for (int i = 0; i < n; i++) {
                multiplier[i] = 1;
            }
        }

        public Matrix(double[][] m) {
            n = m.length;
            a = m;
        }

        public void randomFill(int min, int max) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    a[i][j] = random.nextInt(max - min + 1) + min;
                }
            }
        }

        public void diagonalFill(int min, int max) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (i == j) {
                        a[i][j] = random.nextInt(max - min + 1) + min;
                    } else {
                        a[i][j] = 0;
                    }
                }
            }
        }

        public void hilbertFill() {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    a[i][j] = 1.0 / (i + j + 1.0);
                }
            }
        }

        public void diagonalDominanceFill(int min, int max, int dominanceKoef) {
            for (int i = 0; i < n; i++) {
                double sum = 0;
                for (int j = 0; j < n; j++) {
                    a[i][j] = random.nextInt(max - min + 1) + min;
                    sum += Math.abs(a[i][j]);
                }
                a[i][i] = (1 - 2 * random.nextInt(2)) * (dominanceKoef * sum + 1);
            }
        }

        @Override
        public Matrix clone() {
            Matrix matrix = new Matrix(n);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    matrix.a[i][j] = this.a[i][j];
                }
            }
            return matrix;
        }

        public void print() {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    System.out.format("%10g\t", a[i][j]);
                }
                System.out.print("\n");
            }
        }

        public double getNorm() {
            if (norm < 0) {
                for (int i = 0; i < n; ++i) {
                    double sum = 0;
                    for (int j = 0; j < n; ++j) {
                        sum += Math.abs(a[i][j]);
                    }
                    if (norm < sum) {
                        norm = sum;
                    }
                }
            }
            return norm;
        }

        public double[][] getMatrixCopy() {
            double[][] result = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    result[i][j] = a[i][j];
                }
            }
            return result;
        }

        public boolean isDiagonalDominant() {
            for (int i = 0; i < n; ++i) {
                double sum = 0;
                for (int j = 0; j < i; ++j) {
                    sum += Math.abs(a[i][j]);
                }
                for (int j = i + 1; j < n; j++) {
                    sum += Math.abs(a[i][j]);
                }
                if (sum >= Math.abs(a[i][i])) {
                    return false;
                }
            }
            return true;
        }

        public double[] transform(double[] vector) {
            double[] result = new double[n];
            for (int i = 0; i < n; i++) {
                result[i] = 0;
                for (int j = 0; j < n; j++) {
                    result[i] += a[i][j] * vector[j];
                }
            }
            return result;
        }

        public double[] transposeTransform(double[] vector) {
            double[] result = new double[n];
            for (int i = 0; i < n; i++) {
                result[i] = 0;
                for (int j = 0; j < n; j++) {
                    result[i] += a[j][i] * vector[j];
                }
            }
            return result;
        }

        public double[] g(double[] b, double[] vector) {
            double[] result = transform(vector);
            for (int i = 0; i < n; i++) {
                result[i] -= b[i];
            }
            return result;
        }

        public double[] jacobiMethod(double[] b, long maxIterations, double epsilon,
                                     boolean zeidelMod, double relaxation) throws InconsistentInputException {
            double[][] b1 = new double[n][n];
            double[][] b2 = new double[n][n];

            double[] d = new double[n];

            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (i != j && (!zeidelMod || i > j)) {
                        b1[i][j] = -a[i][j] / a[i][i];
                    } else if (i < j) {
                        b2[i][j] = -a[i][j] / a[i][i];
                    }
                }
                d[i] = b[i] / a[i][i];
            }

            double major;

            /** Check consistency **/
            if (!zeidelMod) {
                double q = new Matrix(b1).getNorm();

                major = epsilon * (1 - q) / q;

                if (q >= 1) {
                    throw new InconsistentInputException(String.format("Inconsistent: Q = %10f >= 1\n", q));
                }
                if (!isDiagonalDominant()) {
                    throw new InconsistentInputException("No diagonal dominance\n");
                }
            } else {
                double q1 = new Matrix(b1).getNorm();
                double q2 = new Matrix(b2).getNorm();
                double q = (1 - q1) / q2;
                major = epsilon * q;

                if (q1 + q2 >= 1) {
                    throw new InconsistentInputException("Inconsistent: ||B1|| + ||B2|| >= 1 \n");
                }
            }

            int prev = 0;
            double[][] x = new double[2][n];
            for (int i = 0; i < n; ++i) {
                x[prev][i] = d[i];
            }

            for (long k = 0; k < maxIterations; ++k) {
                int next = (prev + 1) % 2;
                for (int i = 0; i < n; ++i) {
                    double value = d[i];
                    for (int j = 0; j < i; ++j) {
                        value += b1[i][j] * x[next][j];
                    }
                    for (int j = i + 1; j < n; ++j) {
                        value += b2[i][j] * x[prev][j];
                    }
                    x[next][i] = value;
                }
                double max = Math.abs(x[next][0] = x[prev][0]);
                for (int i = 0; i < n; i++) {
                    x[next][i] = relaxation * x[next][i] + (1 - relaxation) * x[prev][i];
                    if (Math.abs(x[next][i] - x[prev][i]) > max) {
                        max = Math.abs(x[next][i] - x[prev][i]);
                    }
                }
                prev = next;
                if (max < major) break;
            }

            return x[prev];
        }

        public double[] gaussMethod(double[] vector) {
            double[][] a = getMatrixCopy();
            double[] b = vector.clone();
            int[] order = new int[n];
            for (int i = 0; i < n; i++) {
                order[i] = i;
            }
            for (int k = 0; k < n; ++k) {
                /** find pivot row and pivot column**/
                int maxRow = k, maxColumn = k;
                for (int i = k; i < n; ++i) {
                    for (int j = k; j < n; ++j) {
                        if (Math.abs(a[i][j]) > Math.abs(a[maxRow][maxColumn])) {
                            maxRow = i;
                            maxColumn = j;
                        }
                    }
                }

                /** swap row in A matrix **/
                double[] temp = a[k];
                a[k] = a[maxRow];
                a[maxRow] = temp;

                /** swap column in A matrix **/
                double tmp;
                for (int i = 0; i < n; i++) {
                    tmp = a[i][maxColumn];
                    a[i][maxColumn] = a[i][k];
                    a[i][k] = tmp;
                }

                /** swap rows in answer **/
                int itmp = order[maxColumn];
                order[maxColumn] = order[k];
                order[k] = itmp;

                /** swap corresponding values in constants matrix **/
                double t = b[k];
                b[k] = b[maxRow];
                b[maxRow] = t;

                /** pivot within A and B **/
                for (int i = k + 1; i < n; ++i) {
                    double factor = a[i][k] / a[k][k];
                    b[i] -= factor * b[k];
                    for (int j = k; j < n; ++j) {
                        a[i][j] -= factor * a[k][j];
                    }
                }
            }

            double[] v = new double[n];
            for (int i = n - 1; i >= 0; --i) {
                double sum = 0.0;
                for (int j = i + 1; j < n; ++j) {
                    sum += a[i][j] * v[j];
                }
                v[i] = (b[i] - sum) / a[i][i];
            }
            double[] solution = new double[n];
            for (int i = 0; i < n; i++) {
                solution[order[i]] = v[i];
            }
            return solution;
        }

        private double[] conjugateGradientsMethod(double[] b, long r) {
            double[][] sym = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    for (int k = 0; k < n; k++) {
                        sym[i][j] += a[k][i] * a[k][j];
                    }
                }
            }
            Matrix m = new Matrix(sym);
            b = transposeTransform(b);
            double[] x = new double[n];
            for (int i = 0; i < n; i++) {
                x[i] = random.nextDouble();
            }
            double[] p = m.g(b, x);
            for (int i = 0; i < n; i++) {
                p[i] *= -1;
            }
            /** With precise float operations it should give correct answer after n iterations.
             *  However, we take r rounds of n iterations to be sure. */
            for (int t = 0; t < n * r; t++) {
                double[] Ap = m.transform(p);
                double Apxp = scalarProduct(Ap, p);
                double[] g = m.g(b, x);
                double alpha = -scalarProduct(g, p) / Apxp;
                for (int i = 0; i < n; i++) {
                    x[i] += alpha * p[i];
                }
                g = m.g(b, x);
                double beta = scalarProduct(Ap, g) / Apxp;
                for (int i = 0; i < n; i++) {
                    p[i] = -g[i] + beta * p[i];
                }
            }
            return x;
        }

    }

    private static void printArray(double[] array) {

        for (int i = 0; i < array.length; ++i) {
            System.out.format("%10g\t", array[i]);
        }
        System.out.print("\n");
    }

    private static double scalarProduct(double[] a, double[] b) {
        double result = 0;
        for (int i = 0; i < a.length; i++) {
            result += a[i] * b[i];
        }
        return result;
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        //double[][] m = {{2, 1, 1}, {1, -1, 0}, {3, -1, 2}};
        Matrix matrix = new Matrix(10);
        matrix.diagonalDominanceFill(-10, 10, 2);
        double[] b = {2, -2, 2, 1, 3, 5, 10, 1, 1, -3};

        System.out.println("=== Matrix ===");
        matrix.print();
        System.out.println("=== Gauss method ===");
        printArray(matrix.gaussMethod(b));
        System.out.println("=== Jacobi method ===");
        try {
            printArray(matrix.jacobiMethod(b, 100000L, 0.0001, false, 1));
        } catch (InconsistentInputException e) {
            System.out.println(e.getMessage());
        }
        System.out.println("=== Seidel method ===");
        try {
            printArray(matrix.jacobiMethod(b, 100000L, 0.0001, true, 1));
        } catch (InconsistentInputException e) {
            System.out.println(e.getMessage());
        }
        System.out.println("=== Method of successive over-relaxation ===");
        try {
            printArray(matrix.jacobiMethod(b, 100000L, 0.0001, true, 0.9));
        } catch (InconsistentInputException e) {
            System.out.println(e.getMessage());
        }
        System.out.println("=== Method of conjugate gradients ===");
        printArray(matrix.conjugateGradientsMethod(b, 100));
    }

}
