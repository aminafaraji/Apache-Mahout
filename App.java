package kmeans;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.clustering.conversion.InputDriver;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.utils.clustering.ClusterDumper;

public class App {

	private static final String DIRECTORY_CONTAINING_CONVERTED_INPUT = "Kmeansdata";

	public static void main(String[] args) throws Exception {
		
	// Path to output folder 
	Path output = new Path("home/user/Desktop/Kmeansoutput.txt");
			 
	// Hadoop configuration details
	Configuration conf = new Configuration();
		     HadoopUtil.delete(conf, output);
		     
	run(conf, new Path("/home/user/Desktop/seed.txt"), output, new EuclideanDistanceMeasure(), 3, 0.5, 10);
		}
		
	public static void run(Configuration conf, Path input, Path output, DistanceMeasure measure, int k,
	    	      double convergenceDelta, int maxIterations) throws Exception {

	// Input should be given as sequence file format 
	    	    Path directoryContainingConvertedInput = new Path(output, DIRECTORY_CONTAINING_CONVERTED_INPUT);
	    	    InputDriver.runJob(input, directoryContainingConvertedInput, "org.apache.mahout.math.RandomAccessSparseVector");
	    	 
	    	    // Get initial clusters randomly 
	    	    Path clusters = new Path(output, "random-seeds");
	    	    clusters = RandomSeedGenerator.buildRandom(conf, directoryContainingConvertedInput, clusters, k, measure);
	    	    
	    	    // Run K-means with a given K
	    	    KMeansDriver.run(conf, directoryContainingConvertedInput, clusters, output, convergenceDelta,
	    	        maxIterations, true, 0.0, false);
	    	    
	    	    // run ClusterDumper to display result
	    	    Path outGlob = new Path(output, "clusters-*-final");
	    	    Path clusteredPoints = new Path(output,"clusteredPoints");
	    	
	    	    ClusterDumper clusterDumper = new ClusterDumper(outGlob, clusteredPoints);
	    	    clusterDumper.printClusters(null);
	    	  }

	
}
