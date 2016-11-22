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

  def dynamicMemoryNetwork(numMemorySteps: Int): (String, JValue) = {
    val modelParams: JValue = Models.dynamicMemoryNetworkPlus("positional", 3) merge
    Training.default
    val name = s"dynamic_mn_babi_${numMemorySteps}_memory_layers"
    (name, modelParams)
  }

  def adaptiveDynamicMemoryNetwork(maxMemorySteps: Int, ponderCost: Double): (String, JValue) = {
    val modelParams: JValue = Models.dynamicMemoryNetworkPlus("positional", 3) merge
      Training.default merge
      Models.addAdaptiveComponents(maxMemorySteps, ponderCost)
    val name = s"adaptive_dynamic_mn_babi__${ponderCost}_ponder_cost"
    (name, modelParams)
  }

  val models = Seq(dynamicMemoryNetwork(3), adaptiveDynamicMemoryNetwork(3, 0.05))

  def main(args: Array[String]) {
    new BabiEvaluator(Some("babi_experiments"), models, fileUtil).runPipeline()
  }
}
