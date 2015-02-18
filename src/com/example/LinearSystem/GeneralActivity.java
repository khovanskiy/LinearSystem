package com.example.LinearSystem;

import android.app.ActionBar;
import android.app.Activity;
import android.content.Context;
import android.graphics.Color;
import android.os.AsyncTask;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.*;

import java.util.ArrayList;
import java.util.List;
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

        private double norm = -1;

        public Matrix(int size) {
            n = size;
            a = new double[n][n];
        }

        public Matrix(double[][] m) {
            n = m.length;
            a = m;
        }

        private void solutionsFill(double[] b, int min, int max) {
            int[] solutions = new int[n];
            for (int i = 0; i < solutions.length; ++i) {
                solutions[i] = random.nextInt(max - min + 1) + min;
            }
            for (int i = 0; i < n; ++i) {
                b[i] = 0;
                for (int j = 0; j < n; ++j) {
                    b[i] += a[i][j] * solutions[j];
                }
            }
        }

        public void randomFill(double[] b, int min, int max) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    a[i][j] = random.nextInt(max - min + 1) + min;
                }
            }
            solutionsFill(b, min, max);
        }

        public void diagonalFill(double[] b, int min, int max) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (i == j) {
                        a[i][j] = random.nextInt(max - min + 1) + min;
                    } else {
                        a[i][j] = 0;
                    }
                }
            }
            solutionsFill(b, min, max);
        }

        public void hilbertFill(double[] b, int min, int max) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    a[i][j] = 1.0 / (i + j + 1.0);
                }
            }
            solutionsFill(b, min, max);
        }

        public void diagonalDominanceFill(double[] b, int min, int max, int dominanceKoef) {
            for (int i = 0; i < n; i++) {
                double sum = 0;
                for (int j = 0; j < n; j++) {
                    a[i][j] = random.nextInt(max - min + 1) + min;
                    sum += Math.abs(a[i][j]);
                }
                a[i][i] = (1 - 2 * random.nextInt(2)) * (dominanceKoef * sum + 1);
            }
            solutionsFill(b, min, max);
        }

        @Override
        public Matrix clone() {
            Matrix matrix = new Matrix(n);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    matrix.a[i][j] = this.a[i][j];
                }
            }
            return matrix;
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
                for (int j = 0; j < i; j++) {
                    b2[i][j] = -a[i][j] / a[i][i];
                }
                if (zeidelMod) {
                    for (int j = i + 1; j < n; j++) {
                        b1[i][j] = -a[i][j] / a[i][i];
                    }
                } else {
                    for (int j = i + 1; j < n; j++) {
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
                    throw new InconsistentInputException(String.format("Inconsistent: ||B|| = %10f >= 1\n", q));
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
                x[prev][i] = random.nextDouble();
            }

            for (long k = 0; k < maxIterations; ++k) {
                int next = (prev + 1) % 2;
                for (int i = 0; i < n; ++i) {
                    double value = d[i];
                    for (int j = 0; j < n; ++j) {
                        value += b1[i][j] * x[next][j] + b2[i][j] * x[prev][j];
                    }
                    x[next][i] = value;
                }
                double max = Math.abs(x[next][0] - x[prev][0]);
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

        private double[] conjugateGradientsMethod(double[] b, long r) throws InconsistentInputException{
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
                if (Apxp == 0) break;
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

        public double get(int i, int j) {
            return this.a[i][j];
        }
    }

    private static double scalarProduct(double[] a, double[] b) {
        double result = 0;
        for (int i = 0; i < a.length; i++) {
            result += a[i] * b[i];
        }
        return result;
    }

    private void updateView(Matrix matrix, double[] vector) {
        TextView rangeMinTextView = (TextView) findViewById(R.id.rangeMin);
        rangeMin = Integer.parseInt(rangeMinTextView.getText().toString());

        TextView rangeMaxTextView = (TextView) findViewById(R.id.rangeMax);
        rangeMax = Integer.parseInt(rangeMaxTextView.getText().toString());

        TextView epsilonTextView = (TextView) findViewById(R.id.epsilon);
        epsilon = Double.parseDouble(epsilonTextView.getText().toString());

        TextView maxInterationsTextView = (TextView) findViewById(R.id.maxIterations);
        maxIterations = Long.parseLong(maxInterationsTextView.getText().toString());

        TextView relaxationTextView = (TextView) findViewById(R.id.relaxation);
        relaxation = Double.parseDouble(relaxationTextView.getText().toString());

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
                text.setText(String.format("%.02f", matrix.get(i, j)));
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

    private class MethodResult {
        private boolean success;
        private String name;
        private String errorMessage;
        private double[] vector;

        public MethodResult(String name, double[] result) {
            this.name = name;
            this.vector = result;
            this.success = true;
            this.name = name;
        }

        public MethodResult(String name, String errorMessage) {
            this.name = name;
            this.errorMessage = errorMessage;
            this.success = false;
        }

        public boolean isSuccessed() {
            return success;
        }

        public String getName() {
            return name;
        }

        public double[] getVector() {
            return vector;
        }

        public String getErrorMessage() {
            return errorMessage;
        }
    }

    private class Runner extends AsyncTask<Void, MethodResult, Void> {
        private Matrix matrix;
        private double[] vector;

        public Runner(Matrix matrix, double[] vector) {
            this.matrix = matrix.clone();
            this.vector = vector.clone();
        }

        @Override
        protected void onProgressUpdate(MethodResult... values) {
            if (isCancelled()) {
                return;
            }
            ListView listView = (ListView) findViewById(R.id.methods);
            ArrayAdapter<MethodResult> adapter = (ArrayAdapter<MethodResult>) listView.getAdapter();
            adapter.add(values[0]);
            adapter.notifyDataSetChanged();
        }

        @Override
        protected Void doInBackground(Void... params) {
            publishProgress(new MethodResult("Gauss", matrix.gaussMethod(vector)));
            if (isCancelled()) {
                return null;
            }
            try {
                publishProgress(new MethodResult("Jacobi", matrix.jacobiMethod(vector, maxIterations, epsilon, false, 1)));
            } catch (InconsistentInputException e) {
                publishProgress(new MethodResult("Jacobi", e.getMessage()));
            }
            if (isCancelled()) {
                return null;
            }
            try {
                publishProgress(new MethodResult("Seidel", matrix.jacobiMethod(vector, maxIterations, epsilon, true, 1)));
            } catch (InconsistentInputException e) {
                publishProgress(new MethodResult("Seidel", e.getMessage()));
            }
            if (isCancelled()) {
                return null;
            }
            try {
                publishProgress(new MethodResult("Successive over-relaxation", matrix.jacobiMethod(vector, maxIterations, epsilon, true, relaxation)));
            } catch (InconsistentInputException e) {
                publishProgress(new MethodResult("Successive over-relaxation", e.getMessage()));
            }
            if (isCancelled()) {
                return null;
            }
            try {
                publishProgress(new MethodResult("Conjugate gradients", matrix.conjugateGradientsMethod(vector, 1000)));
            } catch (InconsistentInputException e) {
                publishProgress(new MethodResult("Conjugate gradients", e.getMessage()));
            }
            return null;
        }

        @Override
        protected void onPreExecute() {
            ProgressBar progressBar = (ProgressBar) findViewById(R.id.progressBar);
            progressBar.setVisibility(View.VISIBLE);
            ListView listView = (ListView) findViewById(R.id.methods);
            ArrayAdapter<MethodResult> adapter = (ArrayAdapter<MethodResult>) listView.getAdapter();
            if (adapter == null) {
                adapter = new MethodsAdapter(GeneralActivity.this, R.layout.method, new ArrayList<MethodResult>());
                listView.setAdapter(adapter);
            }
            adapter.clear();
            adapter.notifyDataSetChanged();
        }

        @Override
        protected void onPostExecute(Void aVoid) {
            ProgressBar progressBar = (ProgressBar) findViewById(R.id.progressBar);
            progressBar.setVisibility(View.INVISIBLE);
        }
    }

    private class MethodsAdapter extends ArrayAdapter<MethodResult> {

        public MethodsAdapter(Context context, int resource, List<MethodResult> objects) {
            super(context, resource, objects);
        }

        @Override
        public View getView(int position, View convertView, ViewGroup parent) {
            LayoutInflater inflater = (LayoutInflater) getSystemService(Context.LAYOUT_INFLATER_SERVICE);
            View methodView = inflater.inflate(R.layout.method, parent, false);
            TextView methodName = (TextView) methodView.findViewById(R.id.methodName);
            TextView methodResult = (TextView) methodView.findViewById(R.id.methodResult);
            MethodResult method = getItem(position);
            methodName.setText(method.getName());
            if (method.isSuccessed()) {
                methodResult.setText(printArray(method.getVector()));
            } else {
                methodResult.setText(method.getErrorMessage());
            }
            return methodView;
        }

        private String printArray(double[] array) {
            StringBuffer sb = new StringBuffer();
            for (int i = 0; i < array.length; ++i) {
                sb.append(String.format("%10.5f\t", array[i]));
            }
            sb.append("\n");
            return sb.toString();
        }
    }

    private void execute(Matrix matrix, double[] vector) {
        synchronized (this) {
            if (runner != null) {
                runner.cancel(true);
            }
            runner = new Runner(matrix, vector);
            runner.execute();
        }
    }

    private Runner runner = null;
    private int rangeMin = -50;
    private int rangeMax = 50;
    private double epsilon = 0.0001;
    private long maxIterations = 100000L;
    private double relaxation = 0.9;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        final int n = 10;

        final Matrix matrix = new Matrix(n);
        final double[] b = new double[n];

        matrix.diagonalFill(b, 1, 1);
        updateView(matrix, b);

        Button randomButton = (Button) findViewById(R.id.randomButton);
        randomButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                matrix.randomFill(b, rangeMin, rangeMax);
                updateView(matrix, b);
            }
        });

        Button diagonalButton = (Button) findViewById(R.id.diagonalButton);
        diagonalButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                matrix.diagonalFill(b, rangeMin, rangeMax);
                updateView(matrix, b);
            }
        });

        Button diagonalDominanceButton = (Button) findViewById(R.id.diagonalDominanceButton);
        diagonalDominanceButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                matrix.diagonalDominanceFill(b, rangeMin, rangeMax, 2);
                updateView(matrix, b);
            }
        });

        Button hilbertButton = (Button) findViewById(R.id.hilbertButton);
        hilbertButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                matrix.hilbertFill(b, rangeMin, rangeMax);
                updateView(matrix, b);
            }
        });

        Button runButton = (Button) findViewById(R.id.runButton);
        runButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                updateView(matrix, b);
                execute(matrix, b);
            }
        });
        /*System.out.println("=== Matrix ===");
        matrix.print();
        System.out.println("=== Gauss method ===");
        printArray(matrix.gaussMethod(b));
        System.out.println("=== Jacobi method ===");
        try {
            printArray(matrix.jacobiMethod(b, maxIterations, epsilon, false, 1));
        } catch (InconsistentInputException e) {
            System.out.println(e.getMessage());
        }
        System.out.println("=== Seidel method ===");
        try {
            printArray(matrix.jacobiMethod(b, maxIterations, epsilon, true, 1));
        } catch (InconsistentInputException e) {
            System.out.println(e.getMessage());
        }
        System.out.println("=== Method of successive over-relaxation ===");
        try {
            printArray(matrix.jacobiMethod(b, maxIterations, epsilon, true, relaxation));
        } catch (InconsistentInputException e) {
            System.out.println(e.getMessage());
        }
        System.out.println("=== Method of conjugate gradients ===");
        printArray(matrix.conjugateGradientsMethod(b, 100));*/
    }

}
