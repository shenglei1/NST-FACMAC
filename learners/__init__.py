from .cq_learner import CQLearner

from .facmac_learner_soft import FACMACLearnerSoft

from .facmac_sz_learner import FACMACLearner_sz

from .facmac_learner import FACMACLearner
from .facmac_learner_discrete import FACMACDiscreteLearner
from .maddpg_learner import MADDPGLearner
from .maddpg_learner_discrete import MADDPGDiscreteLearner
from .td3_learner import td3Learner
from .td3_learner_noise1 import td3Learner_noise1

REGISTRY = {}
REGISTRY["cq_learner"] = CQLearner
REGISTRY["facmac_learner_soft"] = FACMACLearnerSoft
REGISTRY["facmac_learner"] = FACMACLearner

REGISTRY["facmac_sz_learner"] = FACMACLearner_sz


REGISTRY["facmac_learner_discrete"] = FACMACDiscreteLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["maddpg_learner_discrete"] = MADDPGDiscreteLearner
REGISTRY["td3_learner"] = td3Learner
REGISTRY["td3_learner_noise1"] = td3Learner_noise1