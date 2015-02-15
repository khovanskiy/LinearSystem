package com.example.LinearSystem;

import android.app.Activity;
import android.content.Context;
import android.graphics.Color;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.*;

import java.util.Arrays;
import java.util.Random;

public class GeneralActivity extends Activity {

    private static Random random = new Random();

    private static void solutionsFill(double[][] matrix, double[] b, int min, int max) {
        int[] solutions = new int[matrix.length];
        for (int i = 0; i < solutions.length; ++i) {
            solutions[i] = random.nextInt(max - min + 1) + min;
        }
        for (int i = 0; i < matrix.length; ++i) {
            b[i] = 0;
            for (int j = 0; j < matrix[i].length; ++j) {
                b[i] += matrix[i][j] * solutions[j];
            }
        }
    }

    private static void randomFill(double[][] matrix, double[] b, int min, int max) {
        for (int i = 0; i < matrix.length; ++i) {
            for (int j = 0; j < matrix[i].length; ++j) {
                matrix[i][j] = random.nextInt(max - min + 1) + min;
            }
        }
        solutionsFill(matrix, b, min, max);
    }

    private static void diagonalFill(double[][] matrix, double[] b, int min, int max) {
        for (int i = 0; i < matrix.length; ++i) {
            for (int j = 0; j < matrix[i].length; ++j) {
                if (i == j) {
                    matrix[i][j] = random.nextInt(max - min + 1) + min;
                } else {
                    matrix[i][j] = 0;
                }
            }
        }
        solutionsFill(matrix, b, min, max);
    }

    private static void hilbertFill(double[][] matrix, double[] b, int min, int max) {
        for (int i = 0; i < matrix.length; ++i) {
            for (int j = 0; j < matrix[i].length; ++j) {
                matrix[i][j] = 1.0 / (i + j + 1.0);
            }
        }
        solutionsFill(matrix, b, min, max);
    }

    private static double[][] copyMatrix(double[][] matrix) {
        double[][] temp = new double[matrix.length][];
        for (int i = 0; i < matrix.length; ++i) {
            temp[i] = matrix[i].clone();
        }
        return temp;
    }

    private static void printArray(double[] array) {
        for (int i = 0; i < array.length; ++i) {
            System.out.format("%10g\t", array[i]);
        }
        System.out.print("\n");
    }

    private static void printMatrix(double[][] matrix) {
        for (int i = 0; i < matrix.length; ++i) {
            for (int j = 0; j < matrix[i].length; ++j) {
                System.out.format("%10g\t", matrix[i][j]);
            }
            System.out.print("\n");
        }
    }

    private double matrixNorm(double[][] matrix) {
        double q = -1;
        for (int i = 0; i < matrix.length; ++i) {
            double sum = 0;
            for (int j = 0; j < matrix[i].length; ++j) {
                sum += Math.abs(matrix[i][j]);
            }
            if (q < sum) {
                q = sum;
            }
        }
        return q;
    }

    private double[] seidelMethod(double[][] matrix, double[] b, long maxIterations, double epsilon) {
        double[][] a = copyMatrix(matrix);
        int n = b.length;

        double[][] b1 = new double[n][n];
        double[][] b2 = new double[n][n];
        double[] d = new double[n];

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i > j) {
                    b1[i][j] = -a[i][j] / a[i][i];
                } else if (i < j) {
                    b2[i][j] = -a[i][j] / a[i][i];
                }
            }
            d[i] = b[i] / a[i][i];
        }

        double q1 = matrixNorm(b1);
        double q2 = matrixNorm(b2);
        double q = (1 - q1) / q2;
        double major = epsilon * q;
        if (q1 + q2 >= 1) {
            System.out.print("Inconsistent: ||B1|| + ||B2|| >= 1 \n");
        } else {
            System.out.format("Q = %10f \n", q);
        }

        int current = 0;
        double[][] state = new double[2][n];
        for (int i = 0; i < n; ++i) {
            state[current][i] = d[i];
        }

        for (long k = 0; k < maxIterations; ++k) {
            int next = (current + 1) % 2;
            for (int i = 0; i < n; ++i) {
                double value = d[i];
                for (int j = 0; j < i; ++j) {
                    value += b1[i][j] * state[next][j];
                }
                for (int j = i + 1; j < n; ++j) {
                    value += b2[i][j] * state[current][j];
                }
                state[next][i] = value;
            }
            double max = -1;
            for (int i = 0; i < n; ++i) {
                double t = Math.abs(state[next][i] - state[current][i]);
                if (max < t) {
                    max = t;
                }
            }
            current = next;
            if (max < major) {
                break;
            }
        }

        return state[current];
    }

    private double[] jacobiMethod(double[][] a, double[] b, long maxIterations, double epsilon) {
        int n = b.length;

        double[][] alpha = new double[n][n];
        double[] beta = new double[n];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    alpha[i][j] = -a[i][j] / a[i][i];
                } else {
                    alpha[i][j] = 0;
                }
            }
            beta[i] = b[i] / a[i][i];
        }

        double q = matrixNorm(alpha);

        double major = epsilon * (1 - q) / q;
        boolean dominance = true;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (Math.abs(alpha[i][j]) >= Math.abs(alpha[i][i])) {
                    dominance = false;
                    break;
                }
            }
            if (!dominance) {
                break;
            }
        }
        if (q >= 1) {
            System.out.format("Inconsistent: Q = %10f >= 1\n", q);
        }
        if (!dominance) {
            System.out.format("No diagonal dominance\n");
        }
        if (dominance && q < 1) {
            System.out.format("Q = %10f\n", q);
        }

        int current = 0;
        double[][] state = new double[2][n];
        for (int i = 0; i < n; ++i) {
            state[current][i] = beta[i];
        }

        for (long k = 0; k < maxIterations; ++k) {
            int next = (current + 1) % 2;
            for (int i = 0; i < n; ++i) {
                state[next][i] = beta[i];
                for (int j = 0; j < n; ++j) {
                    state[next][i] += alpha[i][j] * beta[j];
                }
            }
            double max = -1;
            for (int i = 0; i < n; ++i) {
                double t = Math.abs(state[next][i] - state[current][i]);
                if (max < t) {
                    max = t;
                }
            }
            current = next;
            if (max < major) {
                break;
            }
        }

        return state[current];
    }

    private double[] gaussMethod(double[][] matrix, double[] vector) {
        double[][] a = copyMatrix(matrix);
        double[] b = vector.clone();
        int n = b.length;

        for (int k = 0; k < n; ++k) {
            /** find pivot row **/
            int max = k;
            for (int i = k + 1; i < n; ++i) {
                if (Math.abs(a[i][k]) > Math.abs(a[max][k])) {
                    max = i;
                }
            }

            /** swap row in A matrix **/
            double[] temp = a[k];
            a[k] = a[max];
            a[max] = temp;

            /** swap corresponding values in constants matrix **/
            double t = b[k];
            b[k] = b[max];
            b[max] = t;

            /** pivot within A and B **/
            for (int i = k + 1; i < n; ++i) {
                double factor = a[i][k] / a[k][k];
                b[i] -= factor * b[k];
                for (int j = k; j < n; ++j) {
                    a[i][j] -= factor * a[k][j];
                }
            }
        }

        double[] solution = new double[n];
        for (int i = n - 1; i >= 0; --i) {
            double sum = 0.0;
            for (int j = i + 1; j < n; ++j) {
                sum += a[i][j] * solution[j];
            }
            solution[i] = (b[i] - sum) / a[i][i];
        }
        return solution;
    }

    private double[] SORMethod(double[][] matrix, double[] b, long maxIterations, double epsilon) {
        double[][] a = copyMatrix(matrix);
        int n = b.length;

        double[][] b1 = new double[n][n];
        double[][] b2 = new double[n][n];
        double[] d = new double[n];

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i > j) {
                    b1[i][j] = -a[i][j] / a[i][i];
                } else if (i < j) {
                    b2[i][j] = -a[i][j] / a[i][i];
                }
            }
            d[i] = b[i] / a[i][i];
        }

        /*double q1 = matrixNorm(b1);
        double q2 = matrixNorm(b2);
        double q = (1 - q1) / q2;
        double major = epsilon * q;
        if (q1 + q2 >= 1) {
            System.out.print("Inconsistent: ||B1|| + ||B2|| >= 1 \n");
        } else {
            System.out.format("Q = %10f \n", q);
        }*/

        double omega = 0.9;

        int prev = 0;
        double[][] x = new double[2][n];
        for (int i = 0; i < n; ++i) {
            x[prev][i] = d[i];
        }

        for (long k = 0; k < maxIterations; ++k) {
            int next = (prev + 1) % 2;
            for (int i = 0; i < n; ++i) {
                double value = (1 - omega) * x[prev][i] + d[i];
                for (int j = 0; j < i; ++j) {
                    value += b1[i][j] * x[next][j];
                }
                for (int j = i + 1; j < n; ++j) {
                    value += b2[i][j] * x[prev][j];
                }
                x[next][i] = omega * value;
            }
            /*double max = -1;
            for (int i = 0; i < n; ++i) {
                double t = Math.abs(state[next][i] - state[current][i]);
                if (max < t) {
                    max = t;
                }
            }*/
            prev = next;
            /*if (max < major) {
                break;
            }*/
        }

        return x[prev];
    }

    /*private class MatrixAdapter extends BaseAdapter {

        private double[][] matrix;

        public MatrixAdapter(double[][] matrix) {
            this.matrix = matrix;
        }

        @Override
        public int getCount() {
            return this.matrix.length * this.matrix.length;
        }

        @Override
        public Object getItem(int position) {
            return this.matrix[position / matrix.length][position % matrix.length];
        }

        @Override
        public long getItemId(int position) {
            return position;
        }

        @Override
        public View getView(int position, View convertView, ViewGroup parent) {
            LayoutInflater inflater = (LayoutInflater) getSystemService(Context.LAYOUT_INFLATER_SERVICE);
            View view = inflater.inflate(R.layout.element, parent, false);
            TextView text = (TextView) view.findViewById(R.id.textView);
            text.setText(this.matrix[position / matrix.length][position % matrix.length] + "");
            return view;
        }
    }*/

    private void updateView(double[][] matrix, double[] vector) {
        int n = vector.length;
        LinearLayout view = (LinearLayout) findViewById(R.id.lines);
        view.removeAllViewsInLayout();
        for (int i = 0; i < n; ++i) {
            LinearLayout line = new LinearLayout(this);
            line.setOrientation(LinearLayout.HORIZONTAL);
            line.setShowDividers(LinearLayout.SHOW_DIVIDER_MIDDLE);
            line.setLayoutParams(new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));
            for (int j = 0; j < n; ++j) {
                TextView text = new TextView(this);
                text.setText(String.format("%.02f", matrix[i][j]));
                text.setTextColor(Color.parseColor("#ffffffff"));
                text.setLayoutParams(new TableLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
                line.addView(text);
            }
            TextView text = new TextView(this);
            text.setText(String.format("%.02f", vector[i]));
            text.setTextColor(Color.parseColor("#ff0099cc"));
            text.setLayoutParams(new TableLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
            line.addView(text);
            view.addView(line);
        }
        view.invalidate();
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        final int n = 5;
        final int min = -n * 10;
        final int max = n * 10;
        final double epsilon = 0.0001;
        final long maxIterations = 1000000L;

        final double[][] matrix = new double[n][n];
        final double[] b = new double[n];

        diagonalFill(matrix, b, 1, 1);
        updateView(matrix, b);
        //final MatrixAdapter adapter = new MatrixAdapter(matrix);

        Button randomButton = (Button) findViewById(R.id.randomButton);
        randomButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                randomFill(matrix, b, min, max);
                updateView(matrix, b);
            }
        });

        Button diagonalButton = (Button) findViewById(R.id.diagonalButton);
        diagonalButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                diagonalFill(matrix, b, min, max);
                updateView(matrix, b);
            }
        });

        Button hilbertButton = (Button) findViewById(R.id.hilbertButton);
        hilbertButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                hilbertFill(matrix, b, min, max);
                updateView(matrix, b);
            }
        });
        //printMatrix(matrix);
        //diagonalFill(matrix, 0, 10);
        //randomFill(matrix, 0, 10);
        //hilbertFill(matrix);
        System.out.println("=== Matrix ===");
        printMatrix(matrix);
        System.out.println("=== Gauss method ===");
        printArray(gaussMethod(matrix, b));
        System.out.println("=== Jacobi method ===");
        printArray(jacobiMethod(matrix, b, maxIterations, epsilon));
        System.out.println("=== Seidel method ===");
        printArray(seidelMethod(matrix, b, maxIterations, epsilon));
        System.out.println("=== Method of successive over-relaxation ===");
        printArray(SORMethod(matrix, b, maxIterations, epsilon));
    }

}
