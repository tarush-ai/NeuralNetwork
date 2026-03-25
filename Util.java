/*
* Utility for generating binary input files and text truth table files
* consumed by ABCbp. Run this once to produce the data files referenced
* in config.json. Not required at runtime — ABCbp loads files directly.
* Usage: java Util
*/
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.PrintWriter;

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

/*
* Writes a truth table as a plain-text file: one row per case,
* values whitespace-separated. Editable by hand without recompiling.
*/
   public static void saveTruthTable(String filename, double[][] targets) throws Exception
   {
      PrintWriter pw = new PrintWriter(new FileWriter(filename));
      for (double[] row : targets)
      {
         StringBuilder sb = new StringBuilder();
         for (int i = 0; i < row.length; ++i)
         {
            if (i > 0) sb.append(" ");
            sb.append(row[i]);
         }
         pw.println(sb.toString());
      }
      pw.close();
      System.out.println("Saved truth table to: " + filename);
   }

   public static void main(String[] args) throws Exception
   {
/*
* 2-input, 4-case: (0,0) (0,1) (1,0) (1,1)
*/
      double[][] inputs2in4case =
      {
         {0.0, 0.0},
         {0.0, 1.0},
         {1.0, 0.0},
         {1.0, 1.0}
      };

/*
* 3-input, 8-case: all 3-bit combinations
*/
      double[][] inputs3in8case =
      {
         {0.0, 0.0, 0.0},
         {0.0, 0.0, 1.0},
         {0.0, 1.0, 0.0},
         {0.0, 1.0, 1.0},
         {1.0, 0.0, 0.0},
         {1.0, 0.0, 1.0},
         {1.0, 1.0, 0.0},
         {1.0, 1.0, 1.0}
      };

/*
* 3-output AND/OR/XOR truth table for 4 cases (2-input)
*/
      double[][] truth3out4case =
      {
         {0.0, 0.0, 0.0},
         {0.0, 1.0, 1.0},
         {0.0, 1.0, 1.0},
         {1.0, 1.0, 0.0}
      };

/*
* 1-output OR truth table for 4 cases (2-input)
*/
      double[][] truthOr4case =
      {
         {0.0},
         {1.0},
         {1.0},
         {1.0}
      };

/*
* 3-output AND/OR/XOR (parity) truth table for 8 cases (3-input).
* AND: all three inputs are 1.
* OR:  at least one input is 1.
* XOR: odd number of inputs are 1 (parity).
*/
      double[][] truth3out8case =
      {
         {0.0, 0.0, 0.0},
         {0.0, 1.0, 1.0},
         {0.0, 1.0, 1.0},
         {0.0, 1.0, 0.0},
         {0.0, 1.0, 1.0},
         {0.0, 1.0, 0.0},
         {0.0, 1.0, 0.0},
         {1.0, 1.0, 1.0}
      };

      saveInputs("data/inputs/inputs_2in_4case.bin", inputs2in4case);
      saveInputs("data/inputs/inputs_3in_8case.bin", inputs3in8case);
      saveTruthTable("data/truth/truth_andorxor_4case_3out.txt", truth3out4case);
      saveTruthTable("data/truth/truth_or_4case_1out.txt", truthOr4case);
      saveTruthTable("data/truth/truth_andorxor_8case_3out.txt", truth3out8case);
   } // main
} // class Util
