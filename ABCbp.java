/*
* Author: Tarush Gupta
* Date: March 25, 2026
* Description: A-B-C Neural Network with gradient descent training.
*              Generalizes AB1 to support multiple output activations F[i].
*              Supports configurable layers, rand/fixed init, training/running modes,
*              and binary weight save/load.
*/
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.util.function.DoubleUnaryOperator;

public class ABCbp
{
/*
* Network config
*/
   public int inputLayer;
   public int hiddenLayer;
   public int outputLayer;
   public double lambda;
   public double low;
   public double high;
   public int maxIterations;
   public double errorThreshold;
   public boolean useRandomWeights;
   public int numCases;

/*
* Network arrays
*/
   public double[] a;
   public double[] h;
   public double[] F;
   public double[][] wKJ;
   public double[][] wJI;
   public double[][] inputs;
   public double[] thetaJ;

/*
* Run results
*/
   public double[][] results;

/*
* Training-only arrays
*/
   public double[][] targets;
   public double[] psiI;

/*
* Training results
*/
   public int trainIterations;
   public double trainError;
   public long trainTimeMs;
   public long runTimeMs;

/*
* Config file and runtime configuration
*/
   public boolean isTraining;
   public boolean shouldRun;
   public boolean showInputTable;
   public boolean showTruthTable;
   public String loadFile;
   public String saveFile;
   public String inputFile;
   public String truthTableFile;
   public String configFile;
   public String activationFunction;
   public DoubleUnaryOperator activationFn;
   public DoubleUnaryOperator activationDerivFn;

/*
* Reads all network parameters from the given JSON config file.
* Sets activationFn and activationDerivFn via method reference based
* on the activationFunction field — no conditional is evaluated in the loop.
*/
   public void setConfig(String configFile) throws Exception
   {
      this.configFile = configFile;

      StringBuilder sb = new StringBuilder();
      BufferedReader reader = new BufferedReader(new FileReader(configFile));
      String line;
      while ((line = reader.readLine()) != null)
      {
         sb.append(line);
      }
      reader.close();
      String json = sb.toString();

      inputLayer = Integer.parseInt(getJsonValue(json, "inputLayer"));
      hiddenLayer = Integer.parseInt(getJsonValue(json, "hiddenLayer"));
      outputLayer = Integer.parseInt(getJsonValue(json, "outputLayer"));
      numCases = Integer.parseInt(getJsonValue(json, "numCases"));
      inputFile = getJsonValue(json, "inputFile");
      truthTableFile = getJsonValue(json, "truthTableFile");
      isTraining = Boolean.parseBoolean(getJsonValue(json, "isTraining"));
      useRandomWeights = Boolean.parseBoolean(getJsonValue(json, "useRandomWeights"));
      low = Double.parseDouble(getJsonValue(json, "low"));
      high = Double.parseDouble(getJsonValue(json, "high"));
      lambda = Double.parseDouble(getJsonValue(json, "lambda"));
      maxIterations = Integer.parseInt(getJsonValue(json, "maxIterations"));
      errorThreshold = Double.parseDouble(getJsonValue(json, "errorThreshold"));
      shouldRun = Boolean.parseBoolean(getJsonValue(json, "shouldRun"));
      activationFunction = getJsonValue(json, "activationFunction");
      showInputTable = Boolean.parseBoolean(getJsonValue(json, "showInputTable"));
      showTruthTable = Boolean.parseBoolean(getJsonValue(json, "showTruthTable"));
      loadFile = getJsonValue(json, "loadFile");
      saveFile = getJsonValue(json, "saveFile");

      if (activationFunction.equals("tanh"))
      {
         activationFn = this::tanh;
         activationDerivFn = this::tanhDerivative;
      }
      else if (activationFunction.equals("linear"))
      {
         activationFn = this::linear;
         activationDerivFn = this::linearDerivative;
      }
      else
      {
         activationFn = this::sigmoid;
         activationDerivFn = this::sigmoidDerivative;
      }
   } // setConfig(String configFile)

/*
* Extracts the raw value string for the given key from a JSON string.
* Searches for "key": and returns trimmed text up to the next comma or
* closing brace. Returns empty string if the key is not found.
*/
   private String getJsonValue(String json, String key)
   {
      String result = "";
      int keyIdx = json.indexOf("\"" + key + "\"");

      if (keyIdx != -1)
      {
         int colonIdx = json.indexOf(":", keyIdx);
         int commaIdx = json.indexOf(",", colonIdx);
         int braceIdx = json.indexOf("}", colonIdx);
         int endIdx   = (commaIdx != -1 && commaIdx < braceIdx) ? commaIdx : braceIdx;
         result = json.substring(colonIdx + 1, endIdx).trim().replace("\"", "");
      }

      return result;
   } // getJsonValue(String json, String key)

/*
* Prints current network config including file names and training parameters
* when applicable. Only echoes file names for files that will actually be used.
*/
   public void echoConfig()
   {
      System.out.println("Config File: " + configFile);
      System.out.println("Network Configuration: " + inputLayer + "-" + hiddenLayer + "-" + outputLayer);
      System.out.println("Number of Test Cases: " + numCases);
      System.out.println("Input File: " + inputFile);
      System.out.println("Activation Function: " + activationFunction);
      System.out.println("Mode: " + (isTraining ? "Training" : "Running"));
      System.out.println("Run After Training: " + shouldRun);

      if (isTraining)
      {
         System.out.println("\nTraining Parameters:");
         System.out.println("Truth Table File: " + truthTableFile);
         System.out.println("Weight Initialization: " + (useRandomWeights ? "Random" : "Fixed"));

         if (useRandomWeights)
         {
            System.out.println("Random Number Range: [" + low + ", " + high + "]");
         }

         System.out.println("Max Iterations: " + maxIterations);
         System.out.println("Error Threshold: " + errorThreshold);
         System.out.println("Lambda: " + lambda);
      } // if (isTraining)

      if (loadFile != null && loadFile.length() > 0)
      {
         System.out.println("Load File: " + loadFile);
      }

      if (saveFile != null && saveFile.length() > 0)
      {
         System.out.println("Save File: " + saveFile);
      }
   } // echoConfig()

/*
* Allocates memory for all network arrays. Training-only arrays are only
* allocated when the network is in training mode.
*/
   public void allocateMemory()
   {
      a = new double[inputLayer];
      h = new double[hiddenLayer];
      F = new double[outputLayer];
      wKJ = new double[inputLayer][hiddenLayer];
      wJI = new double[hiddenLayer][outputLayer];
      inputs = new double[numCases][inputLayer];

      if (!(isTraining && !shouldRun))
      {
         results = new double[numCases][outputLayer];
      }

      if (isTraining)
      {
         targets = new double[numCases][outputLayer];
         thetaJ = new double[hiddenLayer];
         psiI = new double[outputLayer];
      }
   } // allocateMemory()

/*
* Reads binary input activations from the given file into the inputs array.
*/
   private void loadInputs(String filename) throws Exception
   {
      DataInputStream in = new DataInputStream(new FileInputStream(filename));

      for (int c = 0; c < numCases; ++c)
      {
         for (int k = 0; k < inputLayer; ++k)
         {
            inputs[c][k] = in.readDouble();
         }
      }

      in.close();
   } // loadInputs(String filename)

/*
* Reads a whitespace-delimited text truth table from the given file
* into the targets array. One row per case.
*/
   private void loadTruthTable(String filename) throws Exception
   {
      BufferedReader br = new BufferedReader(new FileReader(filename));

      for (int c = 0; c < numCases; ++c)
      {
         String[] tokens = br.readLine().trim().split("\\s+");
         for (int i = 0; i < outputLayer; ++i)
         {
            targets[c][i] = Double.parseDouble(tokens[i]);
         }
      }

      br.close();
   } // loadTruthTable(String filename)

/*
* Populates network arrays from external files specified in the config.
* Input activations are loaded from binary; truth table from plain text.
* Weight arrays are then initialized (random, loaded from file, or fixed).
*/
   public void populateArrays() throws Exception
   {
      loadInputs(inputFile);

      if (isTraining)
      {
         loadTruthTable(truthTableFile);
      }

      if (useRandomWeights)
      {
         useRandomWeights();
      }
      else if (loadFile != null && loadFile.length() > 0)
      {
         useLoadedWeights();
      }
      else
      {
         useFixedWeights();
      }
   } // populateArrays()

/*
* Prints the input table by iterating through the inputs.
*/
   public void printInputTable()
   {
      System.out.println("\nInput Table:");
      for (int c = 0; c < numCases; ++c)
      {
         for (int k = 0; k < inputLayer; ++k)
         {
            System.out.print(inputs[c][k] + "\t");
         }
         System.out.println();
      }
   } // printInputTable()

/*
* Prints the truth table by iterating through the inputs and targets,
* showing all output columns for each test case.
*/
   public void printTruthTable()
   {
      System.out.println("\nTruth Table:");
      for (int c = 0; c < numCases; ++c)
      {
         for (int k = 0; k < inputLayer; ++k)
         {
            System.out.print(inputs[c][k] + "\t");
         }
         System.out.print("| ");
         for (int i = 0; i < outputLayer; ++i)
         {
            System.out.print(targets[c][i] + "\t");
         }
         System.out.println();
      } // for (int c = 0; c < numCases; ++c)
   } // printTruthTable()

/*
* Initializes all weights randomly using the randomize function.
*/
   public void useRandomWeights()
   {
      for (int k = 0; k < inputLayer; ++k)
      {
         for (int j = 0; j < hiddenLayer; ++j)
         {
            wKJ[k][j] = randomize(low, high);
         }
      }

      for (int j = 0; j < hiddenLayer; ++j)
      {
         for (int i = 0; i < outputLayer; ++i)
         {
            wJI[j][i] = randomize(low, high);
         }
      }
   } // useRandomWeights()

/*
* Uses fixed weights for 2-2-1 testing.
*/
   public void useFixedWeights()
   {
      if (inputLayer == 2 && hiddenLayer == 2 && outputLayer == 1)
      {
         wKJ[0][0] = 0.76;
         wKJ[0][1] = 0.9;
         wKJ[1][0] = 0.9;
         wKJ[1][1] = 0.76;
         wJI[0][0] = 0.2;
         wJI[1][0] = 0.2;
      }
   } // useFixedWeights()

/*
* Uses the weights from the loaded file.
*/
   public void useLoadedWeights()
   {
      try
      {
         DataInputStream in = new DataInputStream(new FileInputStream(loadFile));

         int fileA = in.readInt();
         int fileB = in.readInt();
         int fileC = in.readInt();

         if (fileA == inputLayer && fileB == hiddenLayer && fileC == outputLayer)
         {
            for (int k = 0; k < inputLayer; ++k)
            {
               for (int j = 0; j < hiddenLayer; ++j)
               {
                  wKJ[k][j] = in.readDouble();
               }
            }

            for (int j = 0; j < hiddenLayer; ++j)
            {
               for (int i = 0; i < outputLayer; ++i)
               {
                  wJI[j][i] = in.readDouble();
               }
            }
         } // if (fileA == inputLayer && fileB == hiddenLayer && fileC == outputLayer)

         else
         {
            System.out.println("Error: Weight file config (" + fileA + "-" + fileB + "-" + fileC
               + ") does not match network config ("
               + inputLayer + "-" + hiddenLayer + "-" + outputLayer + ").");
         } // else

         in.close();
      } // try
      catch (Exception e)
      {
         System.out.println("Error loading weights: " + e.getMessage());
      }
   } // useLoadedWeights()

/*
* Saves the weights to the relevant file.
*/
   public void saveWeights()
   {
      try
      {
         DataOutputStream out = new DataOutputStream(new FileOutputStream(saveFile));

         out.writeInt(inputLayer);
         out.writeInt(hiddenLayer);
         out.writeInt(outputLayer);

         for (int k = 0; k < inputLayer; ++k)
         {
            for (int j = 0; j < hiddenLayer; ++j)
            {
               out.writeDouble(wKJ[k][j]);
            }
         }

         for (int j = 0; j < hiddenLayer; ++j)
         {
            for (int i = 0; i < outputLayer; ++i)
            {
               out.writeDouble(wJI[j][i]);
            }
         }

         out.close();
      } // try
      catch (Exception e)
      {
         e.printStackTrace();
      }
   } // saveWeights()

/*
* Returns a random double between low and high. It uses
* Math.random().
*/
   public double randomize(double low, double high)
   {
      return low + Math.random() * (high - low);
   } // randomize(double low, double high)

/*
* Dispatches to the configured activation function (sigmoid, tanh, or linear).
* The function reference is bound once in setConfig(); no conditional is
* evaluated at call time.
*/
   public double activation(double x)
   {
      return activationFn.applyAsDouble(x);
   }

/*
* Dispatches to the configured activation function derivative.
* The function reference is bound once in setConfig(); no conditional is
* evaluated at call time.
*/
   public double activationDerivative(double num)
   {
      return activationDerivFn.applyAsDouble(num);
   }

/*
* Sigmoid activation: f(x) = 1 / (1 + e^-x). Uses Math.exp().
*/
   public double sigmoid(double x)
   {
      return 1.0 / (1.0 + Math.exp(-x));
   } // sigmoid(double x)

/*
* Sigmoid activation derivative: f'(x) = f(x) * (1 - f(x))
*/
   public double sigmoidDerivative(double x)
   {
      double s = sigmoid(x);
      return s * (1.0 - s);
   }

/*
* Hyperbolic tangent activation using the numerically stable epsilon formulation
* from the design document. eps(x) = (x < 0) ? 1 : -1. exp(eps * 2x) is computed
* once and reused in both the numerator and denominator. Math.tanh() is NOT used
* because its behavior for large |x| is not guaranteed across implementations.
*/
   public double tanh(double x)
   {
      double eps = (x < 0) ? 1.0 : -1.0;
      double e = Math.exp(eps * 2.0 * x);
      return eps * (e - 1.0) / (e + 1.0);
   } // tanh(double x)

/*
* Hyperbolic tangent derivative: f'(x) = 1 - tanh^2(x).
* tanh(x) is recomputed directly via the method call rather than using
* any cached activation value.
*/
   public double tanhDerivative(double x)
   {
      double t = tanh(x);
      return 1.0 - t * t;
   } // tanhDerivative(double x)

/*
* Linear activation: f(x) = x
*/
   public double linear(double x)
   {
      return x;
   }

/*
* Linear activation derivative: f'(x) = 1
*/
   public double linearDerivative(double x)
   {
      return 1.0;
   }

/*
* Sets input activations from the given training case and runs the
* network forward. Forward pass: theta_j -> h_j -> theta_i -> F_i
*/
   public double runTrainCase(int caseIndex)
   {
      int i;
      int j;
      int k;

      for (j = 0; j < hiddenLayer; ++j)
      {
         thetaJ[j] = 0.0;
         for (k = 0; k < inputLayer; ++k)
         {
            thetaJ[j] += a[k] * wKJ[k][j];
         }

         h[j] = activation(thetaJ[j]);
      } // for (j = 0; j < hiddenLayer; ++j)

      double twiceError = 0.0;

      for (i = 0; i < outputLayer; ++i)
      {
         double thetaI = 0.0;
         for (j = 0; j < hiddenLayer; ++j)
         {
            thetaI += h[j] * wJI[j][i];
         }

         F[i] = activation(thetaI);

         double omegaI = targets[caseIndex][i] - F[i];

         psiI[i] = omegaI * activationDerivative(thetaI);

         twiceError += omegaI * omegaI;
      } // for (i = 0; i < outputLayer; ++i)

      return twiceError;
   } // runTrainCase(int caseIndex)

/*
* Sets input activations from the given test case and runs the
* network forward. Forward pass: theta_j -> h_j -> theta_i -> F_i
*/
   public void runTestCase()
   {
      int i;
      int j;
      int k;

/*
* Hidden layer: theta_j = sum_K a_K * w_kj, h_j = f(theta_j)
*/
      for (j = 0; j < hiddenLayer; ++j)
      {
         double thetaJ = 0.0;
         for (k = 0; k < inputLayer; ++k)
         {
            thetaJ += a[k] * wKJ[k][j];
         }

         h[j] = activation(thetaJ);
      }

/*
* Output layer: theta_i = sum_J h_J * w_ji, F_i = f(theta_i)
*/
      for (i = 0; i < outputLayer; ++i)
      {
         double thetaI = 0.0;
         for (j = 0; j < hiddenLayer; ++j)
         {
            thetaI += h[j] * wJI[j][i];
         }

         F[i] = activation(thetaI);
      }
   } // runTestCase()

/*
* Trains the network using steepest descent across all training cases.
* Stores final iteration count and error in trainIterations and trainError.
* Backward: omega_i -> psi_i -> Omega_j -> Psi_j
* Gradients: delta_w_ji = lambda * h_j * psi_i, delta_w_kj = lambda * a_k * Psi_j
*/
   public void train()
   {
      int iteration = 0;
      double avgError = 1.0;
      int caseIndex;
      int i;
      int j;
      int k;
      double totalError;

      while (iteration < maxIterations && avgError >= errorThreshold)
      {
         totalError = 0.0;

         for (caseIndex = 0; caseIndex < numCases; ++caseIndex)
         {
            populateInput(caseIndex);
            totalError += runTrainCase(caseIndex);

/*
* Omega_j = sum_I psi_I * w_jI, Psi_j = Omega_j * f'(theta_j)
* delta_w_ji = lambda * h_j * psi_i
* delta_w_kj = lambda * a_k * Psi_j
*/
            for (j = 0; j < hiddenLayer; ++j)
            {
               double omegaJ = 0.0;
               double psiJ = 0.0;

               for (i = 0; i < outputLayer; ++i)
               {
                  omegaJ += psiI[i] * wJI[j][i];
                  double deltaWJI = lambda * h[j] * psiI[i];
                  wJI[j][i] += deltaWJI;
               }

               psiJ = omegaJ * activationDerivative(thetaJ[j]);

               for (k = 0; k < inputLayer; ++k)
               {
                  double deltaWKJ = lambda * a[k] * psiJ;
                  wKJ[k][j] += deltaWKJ;
               }
            } // for (j = 0; j < hiddenLayer; ++j)
         } // for (caseIndex = 0; caseIndex < numCases; ++caseIndex)

         avgError = totalError / (2.0 * (double) numCases); //runTrainCase() returns twice the error
         iteration++;
      } // while (iteration < maxIterations && avgError >= errorThreshold)

      trainIterations = iteration;
      trainError = avgError;
   } // train()

/*
* Reports training exit info: reason, iterations, final error, and time.
*/
   public void reportTrainingResults()
   {
      System.out.println("\nTraining Exit Information:");

      if (trainIterations >= maxIterations)
      {
         System.out.println("Reason: Max iterations reached.");
      }

      if (trainError < errorThreshold)
      {
         System.out.println("Reason: Error threshold reached.");
      }

      System.out.println("Iterations reached: " + trainIterations);
      System.out.println("Average Error reached: " + trainError);
      System.out.println("Time to train: " + trainTimeMs + " ms");
   } // reportTrainingResults()

/*
* Loops over all test cases, setting inputs and running the network.
* Stores each output F[i] in results[caseIndex][i].
*/
   public void runTestCases()
   {
      int caseIndex;
      int i;

      for (caseIndex = 0; caseIndex < numCases; ++caseIndex)
      {
         populateInput(caseIndex);
         runTestCase();

         for (i = 0; i < outputLayer; ++i)
         {
            results[caseIndex][i] = F[i];
         }
      }
   } // runTestCases()

/*
* Sets input activations from the given case and runs the network.
*/
   public void populateInput(int caseIndex)
   {
      for (int k = 0; k < inputLayer; ++k)
      {
         a[k] = inputs[caseIndex][k];
      }
   }

/*
* Reports run results with expected vs actual outputs for all output nodes.
* Displays inputs, expected targets, and actual F[i] values per case.
* Output values are printed to 17 decimal places.
*/
   public void reportRunResults()
   {
      int caseIndex;
      int i;
      int k;

      if (showInputTable)
      {
         System.out.print("\nInputs");
         System.out.print("\t\t");
      }
      if (showTruthTable && isTraining)
      {
         System.out.print("| Expected\t");
      }
      System.out.println("| Actual");

      for (caseIndex = 0; caseIndex < numCases; ++caseIndex)
      {
         if (showInputTable)
         {
            for (k = 0; k < inputLayer; ++k)
            {
               System.out.print(inputs[caseIndex][k] + "\t");
            }
         }

         if (showTruthTable && isTraining)
         {
            System.out.print("| ");
            for (i = 0; i < outputLayer; ++i)
            {
               System.out.print(targets[caseIndex][i] + "\t");
            }
         }

         System.out.print("| ");
         for (i = 0; i < outputLayer; ++i)
         {
            System.out.printf("%.17f\t", results[caseIndex][i]);
         }
         System.out.println();
      } // for (caseIndex = 0; caseIndex < numCases; ++caseIndex)

      System.out.println("Time to run: " + runTimeMs + " ms");
   } // reportRunResults()
} // class ABCbp
