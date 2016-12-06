package org.allenai.deep_qa.experiments

import org.allenai.deep_qa.experiments.datasets.BabiDatasets
import org.allenai.deep_qa.experiments.datasets.ScienceDatasets
import org.allenai.deep_qa.pipeline.BabiEvaluator

import com.mattg.util.FileUtil
import org.json4s.JsonDSL._
import org.json4s._

object BabiExperiments {
  val fileUtil = new FileUtil

  def babiMemN2N(numMemoryLayers: Int): (String, JValue) = {
    val modelParams: JValue = Models.endToEndMemoryNetwork("bow", numMemoryLayers) merge Training.long
    val name = s"babi_memn2n_${numMemoryLayers}_layers"
    (name, modelParams)
  }
  def adaptiveBabiMemN2N(maxMemorySteps: Int, ponderCost: Double): (String, JValue) = {
    val modelParams: JValue = Models.adaptiveEndToEndMemoryNetwork("bow", maxMemorySteps, ponderCost) merge
    Training.long
    val name = s"adaptive_babi_memn2n_${ponderCost}_ponder_cost"
    (name, modelParams)
  }

  val models = Seq(babiMemN2N(1), babiMemN2N(3), adaptiveBabiMemN2N(3, 0.05), adaptiveBabiMemN2N(8,0.05))

  def main(args: Array[String]) {
    new BabiEvaluator(Some("babi_experiments"), models, fileUtil).runPipeline()
  }
}
