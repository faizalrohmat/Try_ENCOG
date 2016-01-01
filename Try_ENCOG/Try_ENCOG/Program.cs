using Encog.Engine.Network.Activation;
using Encog.ML.Data.Basic;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training.Lma;
using Encog.Neural.Networks.Training.Propagation.Back;
using Encog.Neural.Networks.Training.Propagation.Manhattan;
using Encog.Neural.Networks.Training.Propagation.Quick;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using Encog.Neural.Networks.Training.Propagation.SCG;
using Encog.Util.Arrayutil;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Try_ENCOG
{
    class Program
    {
        static void Main(string[] args)
        {
            double error = 0.00001;

            double[][] XOR_Input = 
            {
                new[] {0.0,0.0},
                new[] {1.0,0.0},
                new[] {0.0,1.0},
                new[] {1.0,1.0}
            };

            double[][] XOR_Ideal =
            {
                new[] {0.0},
                new[] {1.0},
                new[] {1.0},
                new[] {0.0}
            };

            var trainingSet = new BasicMLDataSet(XOR_Input, XOR_Ideal);

            BasicNetwork network = CreateNetwork();

            //var train = new Backpropagation(network, trainingSet, 0.7, 0.2);
            //var train = new ManhattanPropagation(network, trainingSet, 0.001);
            // var train = new QuickPropagation(network, trainingSet, 2.0);
            //var train = new ResilientPropagation(network, trainingSet);
            //var train = new ScaledConjugateGradient(network, trainingSet);
            var train = new LevenbergMarquardtTraining(network, trainingSet);
            
            int epoch = 0;
            do
            {
                train.Iteration();
                Console.WriteLine("Iteration No: {0}, Error: {1}", ++epoch, train.Error);
            }
            while (train.Error > error);

            foreach (var item in trainingSet)
            {
                var output = network.Compute(item.Input);
                Console.WriteLine("Input: {0}, {1} \tIdeal: {2} \t Actual: {3}", item.Input[0], item.Input[1], item.Ideal[0], output[0]);
            }

            Console.WriteLine("Training done.");
            Console.WriteLine("press any key to continue");
            Console.ReadLine();

            // normalized value
            var weightNorm = new NormalizedField(NormalizationAction.Normalize, "Weights", 50.0, 40.0, 1.0, -1.0);

            double normalizedValue = weightNorm.Normalize(42.5);
            double denormalizedValue = weightNorm.DeNormalize(normalizedValue);
            Console.WriteLine("Normalized value: {0}", normalizedValue.ToString());
            Console.WriteLine("press any key to continue");
            Console.ReadLine();

            // normalized array
            double[] weights = new double[] { 40.0, 42.5, 43.0, 49.0, 50.0 };
            var weightNormArray = new NormalizeArray();
            weightNormArray.NormalizedHigh = 1.0;
            weightNormArray.NormalizedLow = -1.0;
            double[] normalizedWeights = weightNormArray.Process(weights);

            foreach (var item in normalizedWeights)
            {
                Console.WriteLine("Normalized value: {0}", item.ToString());
            }
            Console.WriteLine("press any key to continue");
            Console.ReadLine();
        }

        private static BasicNetwork CreateNetwork()
        {
            var network = new BasicNetwork();
            network.AddLayer(new BasicLayer(null, true, 2));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 2));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
            network.Structure.FinalizeStructure();
            network.Reset();
            return network;
        }
    }
}
