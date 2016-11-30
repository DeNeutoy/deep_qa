package org.allenai.deep_qa.pipeline

import com.mattg.pipeline.Step
import com.mattg.pipeline.SubprocessStep
import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper
import org.json4s.JsonDSL._
import org.json4s._
import org.json4s.native.JsonMethods.{pretty, render}

import scala.sys.process.Process
import scala.sys.process.ProcessLogger

/**
 * This Step executes code in a subprocess to train a neural network using our DeepQA python
 * library.
 *
 * The basic command that we call is `python src/main/python/run_solver.py [param file]`.  We
 * generate the param file given the parameters passed to this class.
 *
 * We don't use SubprocessStep here, even though we're calling a subprocess, because we need to run
 * some setup commands before actually running the subprocess.
 */
class NeuralNetworkTrainerStep(
  params: JValue,
  fileUtil: FileUtil
) extends Step(Some(params), fileUtil) {
  override val name = "Neural Network Trainer Step"
  implicit val formats = DefaultFormats

  val validParams = Seq(
    "model params",
    "name",
    "dataset",
    "validation dataset"
  )

  // The model hash lets us easily tell if this experiment has been run before - we'll just see if
  // anyone has used the exact same parameters.  However, we still want to allow for human-readable
  // names, if desired.  So we blank out the name before computing the hash.  I would just remove
  // it, but I don't know how to do a non-recursive remove with json4s.
  val getGitCommitSha = "git rev-parse HEAD"
  val exitCode = Process(getGitCommitSha).!(ProcessLogger(
    stdoutLine => {
      // This would add the git sha even if the process returns a
      // non-zero exit code, but we catch that below.
      println(s"Found git sha $stdoutLine. Adding to model hash.")
      params merge ("git sha" -> stdoutLine)
    },
    stderrLine => println(stderrLine)
  ))
  if (exitCode != 0){
    throw new RuntimeException(s"Attempted to get Git sha - returned non-zero exit code.")
  }
  val modelHash = (params merge (("name" -> "removed"): JValue)).hashCode.toHexString
  val modelPrefix = s"/efs/data/dlfa/models/$modelHash/"
  val modelName = JsonHelper.extractWithDefault(params, "name", modelHash)
  val modelParamFile = s"${modelPrefix}model_params.json"
  val logFile = s"${modelPrefix}log.txt"
  val errorLogFile = s"${modelPrefix}log.err"

  val trainDataset = new DatasetStep(params \ "dataset", fileUtil)
  val validationDataset = (params \ "validation dataset") match {
    case JNothing => None
    case jval => Some(new DatasetStep(jval, fileUtil))
  }
  val trainFiles = trainDataset.outputs
  val validationFiles = validationDataset.map(_.outputs).getOrElse(Seq())
  val validationParams = if (validationFiles.size > 0) {
    ("validation_files" -> validationFiles): JValue
  } else {
    JNothing
  }

  val modelParams = (params \ "model params") merge (
    ("model_serialization_prefix" -> modelPrefix) ~
    ("train_files" -> trainFiles)
  ) merge validationParams
  val solverScript = "src/main/python/run_solver.py"


  override def _runStep() {
    logger.info("Writing model params to disk")
    fileUtil.writeContentsToFile(modelParamFile, pretty(render(modelParams)))
    val command = s"""python ${solverScript} ${modelParamFile}"""
    logger.info(s"Running the solver with command: $command")
    val process = Process(command)
    val exitCode = process.!(ProcessLogger(
      stdoutLine => {
        if (stdoutLine.contains("val_acc")) println(stdoutLine)
        fileUtil.writeLinesToFile(logFile, Seq(stdoutLine), true)  // true = append to file
      },
      stderrLine => {
        println(stderrLine)
        fileUtil.writeLinesToFile(errorLogFile, Seq(stderrLine), true)  // true = append to file
      }
    ))
    if (exitCode != 0) {
      // TODO(matt): Check here for a ConfigurationError in the python code, and throw an
      // IllegalStateException instead, which has special meaning to the pipeline code.
      throw new RuntimeException("Subprocess returned non-zero exit code: $exitCode")
    }
  }

  override val inputs: Set[(String, Option[Step])] = {
    val trainInputs = trainFiles.map((_, Some(trainDataset))).toSet
    val validationInputs = validationFiles.map((_, validationDataset)).toSet
    trainInputs ++ validationInputs
  }

  // These three outputs are written by the python code.  The config.json file is a specification
  // of the model layers, so that keras can reconstruct the saved model object / computational
  // graph.  The weights.h5 file is a specification of all of the neural network weights for the
  // model specified by the config.json file.  The data_indexer.pkl file is a pickled DataIndexer,
  // which maps words to indices.
  override val outputs = Set(
    logFile,
    modelPrefix + "_weights.h5",
    modelPrefix + "_data_indexer.pkl",
    modelPrefix + "_config.json"
  )

  override val inProgressFile = modelPrefix + "_in_progress"
  override val paramFile = modelPrefix + "_step_params.json"
}
