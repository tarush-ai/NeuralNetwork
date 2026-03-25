/*
* Utility for generating binary input and truth table files consumed by ABC.
* Run this once to produce the data files specified in config.json.
* Usage: java Util <inputFile> <truthTableFile>
*/
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;

public class Util
{
   public static void saveInputs(String filename, double[][] inputs) throws Exception
   {
      DataOutputStream out = new DataOutputStream(new FileOutputStream(filename));
      for (double[] row : inputs)
         for (double val : row)
            out.writeDouble(val);
      out.close();
      System.out.println("Saved inputs to: " + filename);
   }

   public static void saveTruthTable(String filename, double[][] targets) throws Exception
   {
      DataOutputStream out = new DataOutputStream(new FileOutputStream(filename));
      for (double[] row : targets)
         for (double val : row)
            out.writeDouble(val);
      out.close();
      System.out.println("Saved truth table to: " + filename);
   }

   public static double[][] loadInputs(String filename, int numCases, int inputLayer) throws Exception
   {
      DataInputStream in = new DataInputStream(new FileInputStream(filename));
      double[][] inputs = new double[numCases][inputLayer];
      for (int c = 0; c < numCases; ++c)
         for (int k = 0; k < inputLayer; ++k)
            inputs[c][k] = in.readDouble();
      in.close();
      return inputs;
   }

   public static double[][] loadTruthTable(String filename, int numCases, int outputLayer) throws Exception
   {
      DataInputStream in = new DataInputStream(new FileInputStream(filename));
      double[][] targets = new double[numCases][outputLayer];
      for (int c = 0; c < numCases; ++c)
         for (int i = 0; i < outputLayer; ++i)
            targets[c][i] = in.readDouble();
      in.close();
      return targets;
   }

   public static void main(String[] args) throws Exception
   {
      if (args.length < 2)
      {
         System.out.println("Usage: java Util <inputFile> <truthTableFile>");
         System.exit(1);
      }

      String inputFile = args[0];
      String truthFile = args[1];

/*
* 2-input, 4-case dataset: rows are (0,0), (0,1), (1,0), (1,1)
*/
      double[][] inputs =
      {
         {0.0, 0.0},
         {0.0, 1.0},
         {1.0, 0.0},
         {1.0, 1.0}
      };

/*
* 3-output truth table: AND, OR, XOR
*/
      double[][] targets =
      {
         {0.0, 0.0, 0.0},
         {0.0, 1.0, 1.0},
         {0.0, 1.0, 1.0},
         {1.0, 1.0, 0.0}
      };

      saveInputs(inputFile, inputs);
      saveTruthTable(truthFile, targets);
   } // main
} // class Util
