/** Explicit, removable registrations. Unregistered benchmarks display raw text. */
import type { Presentation } from "./types";
import abstract_reason from "./benchmarks/abstract_reason";
import adaptivestep from "./benchmarks/adaptivestep";
import advprompter from "./benchmarks/advprompter";
import aegis from "./benchmarks/aegis";
import afrieval from "./benchmarks/afrieval";
import afrimedqa from "./benchmarks/afrimedqa";
import agc_bench from "./benchmarks/agc_bench";
import agentdojo from "./benchmarks/agentdojo";
import agentic_review_perturbation from "./benchmarks/agentic_review_perturbation";
import agents_last_exam from "./benchmarks/agents_last_exam";
import agi_elo from "./benchmarks/agi_elo";
import ai2d_test from "./benchmarks/ai2d_test";
import aider from "./benchmarks/aider";
import algotune from "./benchmarks/algotune";
import alignment_faking from "./benchmarks/alignment_faking";
import alpacaeval from "./benchmarks/alpacaeval";
import alpha_sql from "./benchmarks/alpha_sql";
import annotating_errors_wcf from "./benchmarks/annotating_errors_wcf";
import appworld from "./benchmarks/appworld";
import arc_agi_3 from "./benchmarks/arc_agi_3";
import arcagi from "./benchmarks/arcagi";
import atmossci_bench from "./benchmarks/atmossci_bench";
import auditing_sabotage_bench from "./benchmarks/auditing_sabotage_bench";
import autoresearchbench from "./benchmarks/autoresearchbench";
import averimatec from "./benchmarks/averimatec";
import babilong from "./benchmarks/babilong";
import bbq from "./benchmarks/bbq";
import beavertails from "./benchmarks/beavertails";
import bedd_basalt from "./benchmarks/bedd_basalt";
import benger from "./benchmarks/benger";
import bertaqa from "./benchmarks/bertaqa";
import biggen from "./benchmarks/biggen";
import bird_sql from "./benchmarks/bird_sql";
import brace from "./benchmarks/brace";
import braveguard from "./benchmarks/braveguard";
import bridging_gap from "./benchmarks/bridging_gap";
import care_enzymes from "./benchmarks/care_enzymes";
import causal_axioms from "./benchmarks/causal_axioms";
import cellverse from "./benchmarks/cellverse";
import ceobench from "./benchmarks/ceobench";
import ceval from "./benchmarks/ceval";
import chartmuseum from "./benchmarks/chartmuseum";
import chatgpt_drift from "./benchmarks/chatgpt_drift";
import chipbench from "./benchmarks/chipbench";
import clasheval from "./benchmarks/clasheval";
import classroom_ai from "./benchmarks/classroom_ai";
import cmmlu from "./benchmarks/cmmlu";
import coffeebench from "./benchmarks/coffeebench";
import complexbench from "./benchmarks/complexbench";
import confagents from "./benchmarks/confagents";
import corebench from "./benchmarks/corebench";
import critic_discernment_game from "./benchmarks/critic_discernment_game";
import crow from "./benchmarks/crow";
import cruxeval from "./benchmarks/cruxeval";
import csedb from "./benchmarks/csedb";
import cseo_bench from "./benchmarks/cseo_bench";
import das_med_hallucination from "./benchmarks/das_med_hallucination";
import data_juicer2 from "./benchmarks/data_juicer2";
import dataclawbench from "./benchmarks/dataclawbench";
import dave from "./benchmarks/dave";
import dbpa from "./benchmarks/dbpa";
import decodingtrust from "./benchmarks/decodingtrust";
import decompile_bench from "./benchmarks/decompile_bench";
import devbench from "./benchmarks/devbench";
import disco from "./benchmarks/disco";
import donotanswer from "./benchmarks/donotanswer";
import doris_mae from "./benchmarks/doris_mae";
import dpai from "./benchmarks/dpai";
import dqvis from "./benchmarks/dqvis";
import dtap_bench from "./benchmarks/dtap_bench";
import edu_circuit_hw from "./benchmarks/edu_circuit_hw";
import eduguardbench from "./benchmarks/eduguardbench";
import edumath from "./benchmarks/edumath";
import egoschema from "./benchmarks/egoschema";
import ehrflowbench from "./benchmarks/ehrflowbench";
import elicitation_game from "./benchmarks/elicitation_game";
import embodied_agent_interface from "./benchmarks/embodied_agent_interface";
import emoji_attack from "./benchmarks/emoji_attack";
import engdesign from "./benchmarks/engdesign";
import engibench from "./benchmarks/engibench";
import enginemt_qa from "./benchmarks/enginemt_qa";
import erbench from "./benchmarks/erbench";
import evouna from "./benchmarks/evouna";
import exploitgym from "./benchmarks/exploitgym";
import faithcot from "./benchmarks/faithcot";
import felm from "./benchmarks/felm";
import fetv from "./benchmarks/fetv";
import fewshot_ttt_bbh from "./benchmarks/fewshot_ttt_bbh";
import fgbench from "./benchmarks/fgbench";
import financebench from "./benchmarks/financebench";
import find_interp from "./benchmarks/find_interp";
import finegrain_t2i from "./benchmarks/finegrain_t2i";
import frontieror from "./benchmarks/frontieror";
import genai_learning from "./benchmarks/genai_learning";
import ghosts_math from "./benchmarks/ghosts_math";
import goodai_ltm_benchmark from "./benchmarks/goodai_ltm_benchmark";
import haiid from "./benchmarks/haiid";
import hallusionbench from "./benchmarks/hallusionbench";
import harmbench from "./benchmarks/harmbench";
import healthadminbench from "./benchmarks/healthadminbench";
import helm_afr from "./benchmarks/helm_afr";
import helm_airbench from "./benchmarks/helm_airbench";
import helm_anthropic_redteam from "./benchmarks/helm_anthropic_redteam";
import helm_bbq from "./benchmarks/helm_bbq";
import helm_bold from "./benchmarks/helm_bold";
import helm_cleva from "./benchmarks/helm_cleva";
import helm_harmbench from "./benchmarks/helm_harmbench";
import helm_real_toxicity_prompts from "./benchmarks/helm_real_toxicity_prompts";
import helm_simple_safety_tests from "./benchmarks/helm_simple_safety_tests";
import helm_thaiexam from "./benchmarks/helm_thaiexam";
import helm_xstest from "./benchmarks/helm_xstest";
import hivmedqa from "./benchmarks/hivmedqa";
import icpc2_code_selector from "./benchmarks/icpc2_code_selector";
import igakuqa from "./benchmarks/igakuqa";
import igakuqa119 from "./benchmarks/igakuqa119";
import ikp from "./benchmarks/ikp";
import image2struct from "./benchmarks/image2struct";
import imgedit from "./benchmarks/imgedit";
import indeterminacy from "./benchmarks/indeterminacy";
import ineqmath from "./benchmarks/ineqmath";
import infibench from "./benchmarks/infibench";
import interchangeable_token_embeddings from "./benchmarks/interchangeable_token_embeddings";
import intercode from "./benchmarks/intercode";
import jailbreakbench from "./benchmarks/jailbreakbench";
import jetts from "./benchmarks/jetts";
import judgetuning from "./benchmarks/judgetuning";
import katakomba from "./benchmarks/katakomba";
import kernelbench from "./benchmarks/kernelbench";
import kmmlu from "./benchmarks/kmmlu";
import kormedmcqa from "./benchmarks/kormedmcqa";
import kris_bench from "./benchmarks/kris_bench";
import lambda_fp_course from "./benchmarks/lambda_fp_course";
import lawbench from "./benchmarks/lawbench";
import legal_rag_bench from "./benchmarks/legal_rag_bench";
import legalbench from "./benchmarks/legalbench";
import lexeval from "./benchmarks/lexeval";
import lingoly from "./benchmarks/lingoly";
import live_agent_risk from "./benchmarks/live_agent_risk";
import liveaopsbench from "./benchmarks/liveaopsbench";
import livebench from "./benchmarks/livebench";
import llm4ir from "./benchmarks/llm4ir";
import llm_bp_tag from "./benchmarks/llm_bp_tag";
import llm_survey_simulation from "./benchmarks/llm_survey_simulation";
import llm_uncertainty_bench from "./benchmarks/llm_uncertainty_bench";
import llmail_inject from "./benchmarks/llmail_inject";
import llms_lean_formalization from "./benchmarks/llms_lean_formalization";
import lohi_drug_discovery from "./benchmarks/lohi_drug_discovery";
import matharena from "./benchmarks/matharena";
import matharena_platform from "./benchmarks/matharena_platform";
import mathconstraint from "./benchmarks/mathconstraint";
import mathvista_mini from "./benchmarks/mathvista_mini";
import medagentboard from "./benchmarks/medagentboard";
import medhelm_ehr_sql from "./benchmarks/medhelm_ehr_sql";
import medhelm_head_qa from "./benchmarks/medhelm_head_qa";
import medhelm_med_mcqa from "./benchmarks/medhelm_med_mcqa";
import medhelm_med_qa from "./benchmarks/medhelm_med_qa";
import medhelm_medbullets from "./benchmarks/medhelm_medbullets";
import medhelm_medcalc_bench from "./benchmarks/medhelm_medcalc_bench";
import medhelm_medec from "./benchmarks/medhelm_medec";
import medhelm_medhallu from "./benchmarks/medhelm_medhallu";
import medhelm_pubmed_qa from "./benchmarks/medhelm_pubmed_qa";
import medhelm_race_based_med from "./benchmarks/medhelm_race_based_med";
import medwatermark_fws from "./benchmarks/medwatermark_fws";
import mergebench from "./benchmarks/mergebench";
import mind2web from "./benchmarks/mind2web";
import mj_bench from "./benchmarks/mj_bench";
import mlip_arena from "./benchmarks/mlip_arena";
import mlrbench from "./benchmarks/mlrbench";
import mmbench_v11 from "./benchmarks/mmbench_v11";
import mmdocrag from "./benchmarks/mmdocrag";
import mmedbench from "./benchmarks/mmedbench";
import mmlu from "./benchmarks/mmlu";
import mmlupro from "./benchmarks/mmlupro";
import mmmu_dev_val from "./benchmarks/mmmu_dev_val";
import mmoral_bench from "./benchmarks/mmoral_bench";
import mochi from "./benchmarks/mochi";
import moe_cap from "./benchmarks/moe_cap";
import monkey_power_laws from "./benchmarks/monkey_power_laws";
import morphkv from "./benchmarks/morphkv";
import morqa from "./benchmarks/morqa";
import mtbbench from "./benchmarks/mtbbench";
import mtbench from "./benchmarks/mtbench";
import multi_moe from "./benchmarks/multi_moe";
import multimodal_stem_ai from "./benchmarks/multimodal_stem_ai";
import mypcbench from "./benchmarks/mypcbench";
import naep_llm_students from "./benchmarks/naep_llm_students";
import nanobaselib from "./benchmarks/nanobaselib";
import naturalreasoning from "./benchmarks/naturalreasoning";
import naturebench from "./benchmarks/naturebench";
import nester from "./benchmarks/nester";
import nis3d from "./benchmarks/nis3d";
import nyu_ctf_bench from "./benchmarks/nyu_ctf_bench";
import oasst from "./benchmarks/oasst";
import ocrbench_v2 from "./benchmarks/ocrbench_v2";
import ood_prediction from "./benchmarks/ood_prediction";
import openbiorq from "./benchmarks/openbiorq";
import os_harm from "./benchmarks/os_harm";
import perfcodebench from "./benchmarks/perfcodebench";
import perumedqa from "./benchmarks/perumedqa";
import phyblock from "./benchmarks/phyblock";
import pku_saferlhf from "./benchmarks/pku_saferlhf";
import planbench from "./benchmarks/planbench";
import preference_dissection from "./benchmarks/preference_dissection";
import prism from "./benchmarks/prism";
import prm800k from "./benchmarks/prm800k";
import programbench from "./benchmarks/programbench";
import proofaug from "./benchmarks/proofaug";
import proteininvbench from "./benchmarks/proteininvbench";
import prox from "./benchmarks/prox";
import psychosis_bench from "./benchmarks/psychosis_bench";
import putnam_axiom from "./benchmarks/putnam_axiom";
import pxplore from "./benchmarks/pxplore";
import qatch from "./benchmarks/qatch";
import radar from "./benchmarks/radar";
import rakuda from "./benchmarks/rakuda";
import rclicks from "./benchmarks/rclicks";
import real_pocqi from "./benchmarks/real_pocqi";
import real_webagents from "./benchmarks/real_webagents";
import realtimeqa from "./benchmarks/realtimeqa";
import reasoning_gym from "./benchmarks/reasoning_gym";
import refgrader from "./benchmarks/refgrader";
import reliancescope from "./benchmarks/reliancescope";
import researchclawbench from "./benchmarks/researchclawbench";
import researchcodebench from "./benchmarks/researchcodebench";
import revolve from "./benchmarks/revolve";
import rewardbench from "./benchmarks/rewardbench";
import risebench from "./benchmarks/risebench";
import robust_reasoning_benchmark from "./benchmarks/robust_reasoning_benchmark";
import safeagentbench from "./benchmarks/safeagentbench";
import scale_mrt from "./benchmarks/scale_mrt";
import sciarena from "./benchmarks/sciarena";
import scigym from "./benchmarks/scigym";
import sgrades from "./benchmarks/sgrades";
import sib200 from "./benchmarks/sib200";
import sketchjudge from "./benchmarks/sketchjudge";
import software_world_models from "./benchmarks/software_world_models";
import stanford_orb from "./benchmarks/stanford_orb";
import statqa from "./benchmarks/statqa";
import subjective_qa from "./benchmarks/subjective_qa";
import sugarcrepe from "./benchmarks/sugarcrepe";
import summeval from "./benchmarks/summeval";
import sustainable_food from "./benchmarks/sustainable_food";
import svrpbench from "./benchmarks/svrpbench";
import swe_smith from "./benchmarks/swe_smith";
import swebench_live from "./benchmarks/swebench_live";
import swepolybench from "./benchmarks/swepolybench";
import synthpai from "./benchmarks/synthpai";
import tabarena from "./benchmarks/tabarena";
import tau2_bench from "./benchmarks/tau2_bench";
import taubench from "./benchmarks/taubench";
import tengu from "./benchmarks/tengu";
import tensortrust from "./benchmarks/tensortrust";
import theagentcompany from "./benchmarks/theagentcompany";
import threeeed from "./benchmarks/threeeed";
import truthfulqa_mc from "./benchmarks/truthfulqa_mc";
import tulu_human_eval from "./benchmarks/tulu_human_eval";
import tumlu from "./benchmarks/tumlu";
import ultrafeedback from "./benchmarks/ultrafeedback";
import vhelm from "./benchmarks/vhelm";
import visit_bench from "./benchmarks/visit_bench";
import visual_riddles from "./benchmarks/visual_riddles";
import visualwebarena from "./benchmarks/visualwebarena";
import vl_rewardbench from "./benchmarks/vl_rewardbench";
import weavebench from "./benchmarks/weavebench";
import wikihow_agent from "./benchmarks/wikihow_agent";
import wildvision from "./benchmarks/wildvision";
import wmt_mqm from "./benchmarks/wmt_mqm";
import wonderbread from "./benchmarks/wonderbread";
import workarena from "./benchmarks/workarena";
import workbench_revisited from "./benchmarks/workbench_revisited";
import worldcentralbanks from "./benchmarks/worldcentralbanks";
import xstest from "./benchmarks/xstest";

export const presentations: Record<string, Presentation> = {
  abstract_reason,
  adaptivestep,
  advprompter,
  aegis,
  afrieval,
  afrimedqa,
  agc_bench,
  agentdojo,
  agentic_review_perturbation,
  agents_last_exam,
  agi_elo,
  ai2d_test,
  aider,
  algotune,
  alignment_faking,
  alpacaeval,
  alpha_sql,
  annotating_errors_wcf,
  appworld,
  arc_agi_3,
  arcagi,
  atmossci_bench,
  auditing_sabotage_bench,
  autoresearchbench,
  averimatec,
  babilong,
  bbq,
  beavertails,
  bedd_basalt,
  benger,
  bertaqa,
  biggen,
  bird_sql,
  brace,
  braveguard,
  bridging_gap,
  care_enzymes,
  causal_axioms,
  cellverse,
  ceobench,
  ceval,
  chartmuseum,
  chatgpt_drift,
  chipbench,
  clasheval,
  classroom_ai,
  cmmlu,
  coffeebench,
  complexbench,
  confagents,
  corebench,
  critic_discernment_game,
  crow,
  cruxeval,
  csedb,
  cseo_bench,
  das_med_hallucination,
  data_juicer2,
  dataclawbench,
  dave,
  dbpa,
  decodingtrust,
  decompile_bench,
  devbench,
  disco,
  donotanswer,
  doris_mae,
  dpai,
  dqvis,
  dtap_bench,
  edu_circuit_hw,
  eduguardbench,
  edumath,
  egoschema,
  ehrflowbench,
  elicitation_game,
  embodied_agent_interface,
  emoji_attack,
  engdesign,
  engibench,
  enginemt_qa,
  erbench,
  evouna,
  exploitgym,
  faithcot,
  felm,
  fetv,
  fewshot_ttt_bbh,
  fgbench,
  financebench,
  find_interp,
  finegrain_t2i,
  frontieror,
  genai_learning,
  ghosts_math,
  goodai_ltm_benchmark,
  haiid,
  hallusionbench,
  harmbench,
  healthadminbench,
  helm_afr,
  helm_airbench,
  helm_anthropic_redteam,
  helm_bbq,
  helm_bold,
  helm_cleva,
  helm_harmbench,
  helm_real_toxicity_prompts,
  helm_simple_safety_tests,
  helm_thaiexam,
  helm_xstest,
  hivmedqa,
  icpc2_code_selector,
  igakuqa,
  igakuqa119,
  ikp,
  image2struct,
  imgedit,
  indeterminacy,
  ineqmath,
  infibench,
  interchangeable_token_embeddings,
  intercode,
  jailbreakbench,
  jetts,
  judgetuning,
  katakomba,
  kernelbench,
  kmmlu,
  kormedmcqa,
  kris_bench,
  lambda_fp_course,
  lawbench,
  legal_rag_bench,
  legalbench,
  lexeval,
  lingoly,
  live_agent_risk,
  liveaopsbench,
  livebench,
  llm4ir,
  llm_bp_tag,
  llm_survey_simulation,
  llm_uncertainty_bench,
  llmail_inject,
  llms_lean_formalization,
  lohi_drug_discovery,
  matharena,
  matharena_platform,
  mathconstraint,
  mathvista_mini,
  medagentboard,
  medhelm_ehr_sql,
  medhelm_head_qa,
  medhelm_med_mcqa,
  medhelm_med_qa,
  medhelm_medbullets,
  medhelm_medcalc_bench,
  medhelm_medec,
  medhelm_medhallu,
  medhelm_pubmed_qa,
  medhelm_race_based_med,
  medwatermark_fws,
  mergebench,
  mind2web,
  mj_bench,
  mlip_arena,
  mlrbench,
  mmbench_v11,
  mmdocrag,
  mmedbench,
  mmlu,
  mmlupro,
  mmmu_dev_val,
  mmoral_bench,
  mochi,
  moe_cap,
  monkey_power_laws,
  morphkv,
  morqa,
  mtbbench,
  mtbench,
  multi_moe,
  multimodal_stem_ai,
  mypcbench,
  naep_llm_students,
  nanobaselib,
  naturalreasoning,
  naturebench,
  nester,
  nis3d,
  nyu_ctf_bench,
  oasst,
  ocrbench_v2,
  ood_prediction,
  openbiorq,
  os_harm,
  perfcodebench,
  perumedqa,
  phyblock,
  pku_saferlhf,
  planbench,
  preference_dissection,
  prism,
  prm800k,
  programbench,
  proofaug,
  proteininvbench,
  prox,
  psychosis_bench,
  putnam_axiom,
  pxplore,
  qatch,
  radar,
  rakuda,
  rclicks,
  real_pocqi,
  real_webagents,
  realtimeqa,
  reasoning_gym,
  refgrader,
  reliancescope,
  researchclawbench,
  researchcodebench,
  revolve,
  rewardbench,
  risebench,
  robust_reasoning_benchmark,
  safeagentbench,
  scale_mrt,
  sciarena,
  scigym,
  sgrades,
  sib200,
  sketchjudge,
  software_world_models,
  stanford_orb,
  statqa,
  subjective_qa,
  sugarcrepe,
  summeval,
  sustainable_food,
  svrpbench,
  swe_smith,
  swebench_live,
  swepolybench,
  synthpai,
  tabarena,
  tau2_bench,
  taubench,
  tengu,
  tensortrust,
  theagentcompany,
  threeeed,
  truthfulqa_mc,
  tulu_human_eval,
  tumlu,
  ultrafeedback,
  vhelm,
  visit_bench,
  visual_riddles,
  visualwebarena,
  vl_rewardbench,
  weavebench,
  wikihow_agent,
  wildvision,
  wmt_mqm,
  wonderbread,
  workarena,
  workbench_revisited,
  worldcentralbanks,
  xstest,
};
