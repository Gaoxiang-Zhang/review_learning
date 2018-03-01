# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Beam search module.

Beam search takes the top K results from the model, predicts the K results for
each of the previous K result, getting K*K results. Pick the top K results from
K*K results, and start over again until certain number of results are fully
decoded.
"""

from six.moves import xrange
import tensorflow as tf

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_bool('normalize_by_length', True, 'Whether to normalize')


class Hypothesis(object):
  """Defines a hypothesis during beam search."""
  # 保存beam search的中间状态
  def __init__(self, tokens, log_prob, state):
    """Hypothesis constructor.

    Args:
      tokens: start tokens for decoding. （decoding的开始tokens）
      log_prob: log prob of the start tokens, usually 1.（当前的log prob）
      state: decoder initial states.（decoder的state）
    """
    self.tokens = tokens
    self.log_prob = log_prob
    self.state = state

  def Extend(self, token, log_prob, new_state):
    """Extend the hypothesis with result from latest step.
    # 根据输入参数，对当前状态进行扩展
    Args:
      token: latest token from decoding.
      log_prob: log prob of the latest decoded tokens.
      new_state: decoder output state. Fed to the decoder for next step.
    Returns:
      New Hypothesis with the results from latest step.
    """
    return Hypothesis(self.tokens + [token], self.log_prob + log_prob,
                      new_state)

  @property
  def latest_token(self):
    return self.tokens[-1]

  def __str__(self):
    return ('Hypothesis(log prob = %.4f, tokens = %s)' % (self.log_prob,
                                                          self.tokens))


class BeamSearch(object):
  """Beam search."""
  # 进行beam search
  def __init__(self, model, beam_size, start_token, end_token, max_steps):
    """Creates BeamSearch object.
    # 一些参数初始化
    Args:
      model: Seq2SeqAttentionModel. 
      beam_size: int. 
      start_token: int, id of the token to start decoding with （开始的token id）
      end_token: int, id of the token that completes an hypothesis （结束的token id）
      max_steps: int, upper limit on the size of the hypothesis（最大长度）
    """
    self._model = model
    self._beam_size = beam_size
    self._start_token = start_token
    self._end_token = end_token
    self._max_steps = max_steps

  def BeamSearch(self, sess, enc_inputs, enc_seqlen):
    """Performs beam search for decoding.
    # 进行beam search
    Args:
      sess: tf.Session, session
      enc_inputs: ndarray of shape (enc_length, 1), the document ids to encode （encoder的inputs)
      enc_seqlen: ndarray of shape (1), the length of the sequnce （encoder的input len）

    Returns:
      hyps: list of Hypothesis, the best hypotheses found by beam search,
          ordered by score
    """

    # Run the encoder and extract the outputs and final state.
    # 运行encoder，得到encoder的输出和最终状态
    enc_top_states, dec_in_state = self._model.encode_top_state(
        sess, enc_inputs, enc_seqlen)
    # Replicate the initial states K times for the first step.
    # 初始状态，复制beam size次
    hyps = [Hypothesis([self._start_token], 0.0, dec_in_state)
           ] * self._beam_size
    results = []

    steps = 0
    # 进行最大次数为_max_steps的beam search
    while steps < self._max_steps and len(results) < self._beam_size:
      # 得到每个hyps的最后一个词和状态
      latest_tokens = [h.latest_token for h in hyps]
      states = [h.state for h in hyps]
      # decoder运行一步，得到结果
      topk_ids, topk_log_probs, new_states = self._model.decode_topk(
          sess, latest_tokens, enc_top_states, states)
      # Extend each hypothesis.
      all_hyps = []
      # The first step takes the best K results from first hyps. Following
      # steps take the best K results from K*K hyps.
      num_beam_source = 1 if steps == 0 else len(hyps)
      # 对当前的hyps进行扩展
      for i in xrange(num_beam_source):
        h, ns = hyps[i], new_states[i]
        for j in xrange(self._beam_size*2):
          all_hyps.append(h.Extend(topk_ids[i, j], topk_log_probs[i, j], ns))

      # Filter and collect any hypotheses that have the end token.
      # 如果有hyp包含了</s>，则加入results
      hyps = []
      for h in self._BestHyps(all_hyps):
        if h.latest_token == self._end_token:
          # Pull the hypothesis off the beam if the end token is reached.
          results.append(h)
        else:
          # Otherwise continue to the extend the hypothesis.
          hyps.append(h)
        if len(hyps) == self._beam_size or len(results) == self._beam_size:
          break

      steps += 1

    if steps == self._max_steps:
      results.extend(hyps)

    return self._BestHyps(results)

  def _BestHyps(self, hyps):
    """Sort the hyps based on log probs and length.
    # 根据log probs（如果定义了normalization，则还需要考虑len）对hyps进行排序
    Args:
      hyps: A list of hypothesis.
    Returns:
      hyps: A list of sorted hypothesis in reverse log_prob order.
    """
    # This length normalization is only effective for the final results.
    if FLAGS.normalize_by_length:
      return sorted(hyps, key=lambda h: h.log_prob/len(h.tokens), reverse=True)
    else:
      return sorted(hyps, key=lambda h: h.log_prob, reverse=True)
