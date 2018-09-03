#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>
#include <ctime>
#include <unordered_set>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <sstream>
#include <limits>

#include <execinfo.h>
#include <unistd.h>
#include <signal.h>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>

#include "cnn/training.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/nodes.h"
#include "cnn/lstm.h"
#include "cnn/rnn.h"
#include "cnn/dict.h"
#include "cnn/cfsm-builder.h"

#include "impl/oracle.h"
#include "impl/pretrained.h"
#include "impl/compressed-fstream.h"
#include "impl/eval.h"

// dictionaries
cnn::Dict termdict, ntermdict, adict, posdict;
bool DEBUG;
volatile bool requested_stop = false;
unsigned IMPLICIT_REDUCE_AFTER_SHIFT = 0;
unsigned LAYERS = 2;
unsigned INPUT_DIM = 40;
unsigned HIDDEN_DIM = 60;
unsigned ACTION_DIM = 36;
unsigned PRETRAINED_DIM = 50;
unsigned LSTM_INPUT_DIM = 60;
unsigned POS_DIM = 10;

float ALPHA = 1.f;
unsigned N_SAMPLES = 1;
unsigned ACTION_SIZE = 0;
unsigned VOCAB_SIZE = 0;
unsigned NT_SIZE = 0;
float DROPOUT = 0.0f;
unsigned POS_SIZE = 0;
std::map<int,int> action2NTindex;  // pass in index of action PJ(X), return index of X
bool USE_POS = false;  // in discriminative parser, incorporate POS information in token embedding

unsigned CUR_ITER = 0;

using namespace cnn::expr;
using namespace cnn;
using namespace std;
namespace po = boost::program_options;




//DECLARE_____

default_random_engine rng;
uniform_real_distribution<double> uniform(0,1.0);

int num_trans=0;

vector<unsigned> possible_actions;
unordered_map<unsigned, vector<float>> pretrained;
vector<bool> singletons; // used during training


unsigned sample_from(const vector<float>& log_events, float power) {
    float sum = 0.0;
    float sum2 = 0.0;
    vector<float> events;
    for (float e : log_events) { 
        float ee = exp(e); 
        if (ee <= 1e-5) (ee = 0);
        sum+=ee; events.push_back(ee);
    }
    for (int i = 0; i < events.size(); ++i) {
        events[i] = events[i]/sum;
        if (power != 1.0)
            events[i] = pow(events[i],power);
        sum2 += events[i];
        //cout << "e" << events[i] << endl;
    }
    //cout << "   sum:" << sum2 << endl;
    while (1) {
        float target = uniform(rng); 
        //cout << "target:" << target << endl;
        int r = 0;
        for (float e : events) { 
            target -= e / sum2;
            if (target <= 0) {
                //cout << "event:" << e << endl;
                return r;
            }
            r++;
        }
        cout << "beyond end. " << target << ", retrying." << std::endl;
    }
    cout << "beyond end. " << std::endl;
    return 0;
}



void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
        ("training_data,T", po::value<string>(), "List of Transitions - Training corpus")
        ("dev_data,d", po::value<string>(), "Development corpus")
        ("bracketing_dev_data,C", po::value<string>(), "Development bracketed corpus")

        ("test_data,p", po::value<string>(), "Test corpus")
        ("dropout,D", po::value<float>(), "Dropout rate")
        ("samples,s", po::value<unsigned>(), "Sample N trees for each test sentence instead of greedy max decoding")
        ("alpha,a", po::value<float>(), "Flatten (0 < alpha < 1) or sharpen (1 < alpha) sampling distribution")
        ("model,m", po::value<string>(), "Load saved model from this file")
        ("use_pos_tags,P", "make POS tags visible to parser")
        ("layers", po::value<unsigned>()->default_value(2), "number of LSTM layers")
        ("action_dim", po::value<unsigned>()->default_value(16), "action embedding size")
        ("pos_dim", po::value<unsigned>()->default_value(12), "POS dimension")
        ("input_dim", po::value<unsigned>()->default_value(32), "input embedding size")
        ("hidden_dim", po::value<unsigned>()->default_value(64), "hidden dimension")
        ("pretrained_dim", po::value<unsigned>()->default_value(50), "pretrained input dimension")
        ("lstm_input_dim", po::value<unsigned>()->default_value(60), "LSTM input dimension")
        ("train,t", "Should training be run?")
        ("words,w", po::value<string>(), "Pretrained word embeddings")
        ("beam_size,b", po::value<unsigned>()->default_value(1), "beam size")
	("debug","debug")
        ("help,h", "Help");
  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("help")) {
    cerr << dcmdline_options << endl;
    exit(1);
  }
  if (conf->count("training_data") == 0) {
    cerr << "Please specify --traing_data (-T): this is required to determine the vocabulary mapping, even if the parser is used in prediction mode.\n";
    exit(1);
  }
}

struct ParserBuilder {
  LSTMBuilder stack_lstm; // (layers, input, hidden, trainer)
  LSTMBuilder *buffer_lstm;
  LSTMBuilder action_lstm;
  LSTMBuilder const_lstm_fwd;
  LSTMBuilder const_lstm_rev;
  LookupParameters* p_w; // word embeddings
  LookupParameters* p_t; // pretrained word embeddings (not updated)
  LookupParameters* p_nt; // nonterminal embeddings
  LookupParameters* p_ntup; // nonterminal embeddings when used in a composed representation
  LookupParameters* p_a; // input action embeddings
  LookupParameters* p_pos; // pos embeddings (optional)
  Parameters* p_p2w;  // pos2word mapping (optional)
  Parameters* p_ptbias; // preterminal bias (used with IMPLICIT_REDUCE_AFTER_SHIFT)
  Parameters* p_ptW;    // preterminal W (used with IMPLICIT_REDUCE_AFTER_SHIFT)
  Parameters* p_pbias; // parser state bias
  Parameters* p_A; // action lstm to parser state
  Parameters* p_B; // buffer lstm to parser state
  Parameters* p_S; // stack lstm to parser state
  Parameters* p_w2l; // word to LSTM input
  Parameters* p_t2l; // pretrained word embeddings to LSTM input
  Parameters* p_ib; // LSTM input bias
  Parameters* p_cbias; // composition function bias
  Parameters* p_p2a;   // parser state to action
  Parameters* p_action_start;  // action bias
  Parameters* p_abias;  // action bias
  Parameters* p_buffer_guard;  // end of buffer
  Parameters* p_stack_guard;  // end of stack

  Parameters* p_cW;

  explicit ParserBuilder(Model* model, const unordered_map<unsigned, vector<float>>& pretrained) :
      stack_lstm(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, model),
      action_lstm(LAYERS, ACTION_DIM, HIDDEN_DIM, model),
      const_lstm_fwd(LAYERS, LSTM_INPUT_DIM, LSTM_INPUT_DIM, model), // used to compose children of a node into a representation of the node
      const_lstm_rev(LAYERS, LSTM_INPUT_DIM, LSTM_INPUT_DIM, model), // used to compose children of a node into a representation of the node
      p_w(model->add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM})),
      p_t(model->add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM})),
      p_nt(model->add_lookup_parameters(NT_SIZE, {LSTM_INPUT_DIM})),
      p_ntup(model->add_lookup_parameters(NT_SIZE, {LSTM_INPUT_DIM})),
      p_a(model->add_lookup_parameters(ACTION_SIZE, {ACTION_DIM})),
      p_pbias(model->add_parameters({HIDDEN_DIM})),
      p_A(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_B(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_S(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_w2l(model->add_parameters({LSTM_INPUT_DIM, INPUT_DIM})),
      p_ib(model->add_parameters({LSTM_INPUT_DIM})),
      p_cbias(model->add_parameters({LSTM_INPUT_DIM})),
      p_p2a(model->add_parameters({ACTION_SIZE, HIDDEN_DIM})),
      p_action_start(model->add_parameters({ACTION_DIM})),
      p_abias(model->add_parameters({ACTION_SIZE})),

      p_buffer_guard(model->add_parameters({LSTM_INPUT_DIM})),
      p_stack_guard(model->add_parameters({LSTM_INPUT_DIM})),

      p_cW(model->add_parameters({LSTM_INPUT_DIM, LSTM_INPUT_DIM * 2})) {
    if (USE_POS) {
      p_pos = model->add_lookup_parameters(POS_SIZE, {POS_DIM});
      p_p2w = model->add_parameters({LSTM_INPUT_DIM, POS_DIM});
    }
    buffer_lstm = new LSTMBuilder(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, model);
    if (pretrained.size() > 0) {
      p_t = model->add_lookup_parameters(VOCAB_SIZE, {PRETRAINED_DIM});
      for (auto it : pretrained)
        p_t->Initialize(it.first, it.second);
      p_t2l = model->add_parameters({LSTM_INPUT_DIM, PRETRAINED_DIM});
    } else {
      p_t = nullptr;
      p_t2l = nullptr;
    }
  }

// checks to see if a proposed action is valid in discriminative models
static bool IsActionForbidden_Discriminative(const string& a, char prev_a, unsigned bsize, unsigned ssize, unsigned nopen_parens, unsigned unary) {
  bool is_shift = (a[0] == 'S' && a[1]=='H');
  bool is_reduce = (a[0] == 'R' && a[1]=='E');
  bool is_nt = (a[0] == 'N');
  bool is_term = (a[0] == 'T');
  
  assert(is_shift || is_reduce || is_nt || is_term) ;
  static const unsigned MAX_OPEN_NTS = 100;
  static const unsigned MAX_UNARY = 3;
//  if (is_nt && nopen_parens > MAX_OPEN_NTS) return true;
  if (is_term){
	if(ssize == 2 && bsize == 1 && prev_a == 'R') return false;
    return true;
  }

  if(ssize == 1){
     if(!is_shift) return true;
     return false;
  }

  if (is_shift){
    if(bsize == 1) return true;
    if(nopen_parens == 0) return true;
    return false;
  }

  if (is_nt) {
    if(bsize == 1 && unary >= MAX_UNARY) return true;
    if(prev_a == 'N') return true;
    return false; 
  }

  if (is_reduce){
    if(unary > MAX_UNARY) return true;
    if(nopen_parens == 0) return true;
    return false;
  }

  // TODO should we control the depth of the parse in some way? i.e., as long as there
  // are items in the buffer, we can do an NT operation, which could cause trouble
}


// *** if correct_actions is empty, this runs greedy decoding ***
// returns parse actions for input sentence (in training just returns the reference)
// this lets us use pretrained embeddings, when available, for words that were OOV in the
// parser training data
// set sample=true to sample rather than max
vector<unsigned> log_prob_parser(ComputationGraph* hg,
                     const parser::Sentence& sent,
                     const vector<int>& correct_actions,
                     double *right,
                     bool is_evaluation,
                     bool sample = false) {
if(DEBUG) cerr << "********START SENTENCE PARSING****************"<<"\n";
if(DEBUG) cerr << "sent size: " << sent.size()<<"\n";
    vector<unsigned> results;
    const bool build_training_graph = correct_actions.size() > 0;
    bool apply_dropout = (DROPOUT && !is_evaluation);
    stack_lstm.new_graph(*hg);
    action_lstm.new_graph(*hg);
    const_lstm_fwd.new_graph(*hg);
    const_lstm_rev.new_graph(*hg);
    stack_lstm.start_new_sequence();
    buffer_lstm->new_graph(*hg);
    buffer_lstm->start_new_sequence();
    action_lstm.start_new_sequence();
    if (apply_dropout) {
      stack_lstm.set_dropout(DROPOUT);
      action_lstm.set_dropout(DROPOUT);
      buffer_lstm->set_dropout(DROPOUT);
      const_lstm_fwd.set_dropout(DROPOUT);
      const_lstm_rev.set_dropout(DROPOUT);
    } else {
      stack_lstm.disable_dropout();
      action_lstm.disable_dropout();
      buffer_lstm->disable_dropout();
      const_lstm_fwd.disable_dropout();
      const_lstm_rev.disable_dropout();
    }
    // variables in the computation graph representing the parameters
    Expression pbias = parameter(*hg, p_pbias);
    Expression S = parameter(*hg, p_S);
    Expression B = parameter(*hg, p_B);
    Expression A = parameter(*hg, p_A);
    Expression p2w;
    if (USE_POS) {
      p2w = parameter(*hg, p_p2w);
    }

    Expression ib = parameter(*hg, p_ib);
    Expression cbias = parameter(*hg, p_cbias);
    Expression w2l = parameter(*hg, p_w2l);
    Expression t2l;
    if (p_t2l)
      t2l = parameter(*hg, p_t2l);
    Expression p2a = parameter(*hg, p_p2a);
    Expression abias = parameter(*hg, p_abias);
    Expression action_start = parameter(*hg, p_action_start);
    Expression cW = parameter(*hg, p_cW);

    action_lstm.add_input(action_start);

    vector<Expression> buffer(sent.size() + 1);  // variables representing word embeddings
    vector<int> bufferi(sent.size() + 1);  // position of the words in the sentence
    // precompute buffer representation from left to right

    // in the discriminative model, here we set up the buffer contents
    for (unsigned i = 0; i < sent.size(); ++i) {
        int wordid = sent.raw[i]; // this will be equal to unk at dev/test
        if (build_training_graph && singletons.size() > wordid && singletons[wordid] && rand01() > 0.5)
          wordid = sent.unk[i];
        Expression w = lookup(*hg, p_w, wordid);
        vector<Expression> args = {ib, w2l, w}; // learn embeddings
        if (p_t && pretrained.count(sent.lc[i])) {  // include fixed pretrained vectors?
          Expression t = const_lookup(*hg, p_t, sent.lc[i]);
          args.push_back(t2l);
          args.push_back(t);
        }
        if (USE_POS) {
          args.push_back(p2w);
          args.push_back(lookup(*hg, p_pos, sent.pos[i]));
        }
        buffer[sent.size() - i] = rectify(affine_transform(args));
        bufferi[sent.size() - i] = i;
    }
    // dummy symbol to represent the empty buffer
    buffer[0] = parameter(*hg, p_buffer_guard);
    bufferi[0] = -999;
    
    for (auto& b : buffer)
      buffer_lstm->add_input(b);

    vector<Expression> stack;  // variables representing subtree embeddings
    vector<int> stacki; // position of words in the sentence of head of subtree
    stack.push_back(parameter(*hg, p_stack_guard));
    stacki.push_back(-999); // not used for anything
    // drive dummy symbol on stack through LSTM
    stack_lstm.add_input(stack.back());
    vector<int> is_open_paren; // -1 if no nonterminal has a parenthesis open, otherwise index of NT
    is_open_paren.push_back(-1); // corresponds to dummy symbol
    vector<Expression> log_probs;
    string rootword;
    unsigned action_count = 0;  // incremented at each prediction
    unsigned nt_count = 0; // number of times an NT has been introduced
    vector<unsigned> current_valid_actions;
    unsigned unary = 0;
    int nopen_parens = 0;
    char prev_a = '0';
    

         vector<string> gold_constituents;
         vector<string> gold_init;
         vector<string> gold_unique_init;
         vector<int> gold_arity_init;
              
        if (build_training_graph){
        	       	
        		vector<string> gstack; 
        		int index_buffer=0;
        		int index=0;
        				unsigned accion = correct_actions[index];
        				string aGold=adict.Convert(accion);
        		while(aGold[0] != 'T'){
        			
        			if(aGold[0] == 'S'){
        				
        				int bg=index_buffer;
        				index_buffer++;
        				int en=index_buffer;
        				string elto=std::to_string(bg)+"#"+std::to_string(en);
        				gstack.push_back(elto);
        				
        				
        				
        			}else if(aGold[0] == 'N')
        			{
        				string bg=std::to_string(index_buffer);
        				const string& NT = aGold.substr(3,aGold.length()-4);
        				
        				string nt=NT+"#"+bg;
        				
        				    				int pos=contain(gold_unique_init,NT);
        				    				int order=1;
        				    				if(pos!=-1)
        				    				{
        				    					gold_arity_init[pos]=gold_arity_init[pos]+1;
        				    					order=gold_arity_init[pos];
        				    				}else{
        				    					gold_unique_init.push_back(NT);
        				    					gold_arity_init.push_back(int(1));
        				    				}
        				
        				    				gold_init.push_back(nt+"%"+std::to_string(order));
        				    				gstack.push_back("N"+nt+"%"+std::to_string(order));
        				
        			}else if(aGold[0] == 'R')
        			{
           				const string& delimiterA = "#";
            				const string& delimiterB = "%";
            				string NT;
            				string bg;
            				string en=std::to_string(index_buffer);
            				string order="100";
            				while(!gstack.empty())
            				{
            					string elto=gstack.back();
            					gstack.pop_back();
            					if(elto[0]=='N')
            					{
            						string use1 = elto.substr(1);
            						string use = use1.substr(0,use1.find(delimiterB));//Obviamos el %num
            						order= use1.substr(use1.find(delimiterB)+1);
            						NT = use.substr(0,use.find(delimiterA));
//            						bg = use.substr(use.find(delimiterA)+1);
            						string elto2=gstack.back();
            						gstack.pop_back();
            						bg = elto2.substr(0,elto2.find(delimiterA));
            						
            						break;
            					}
            					
            				}
            				
            				
            				
            				
            				gstack.push_back(bg+"#"+en);
            				string nt=NT+"|"+bg+"#"+en+"%"+order;
            				

            				gold_constituents.push_back(nt);
            				
        			}
        			index++;
        			accion = correct_actions[index];
        			aGold=adict.Convert(accion);
        			
        			 
        			
        		}
        	
        }	

        vector<string> predicted_constituents;
            	vector<string> predicted_init;
            	vector<string> predicted_unique_init;
            	     vector<int> predicted_arity_init;
                    vector<string> pstack;
            	int pindex_buffer=0;
        //exit(0);

        //****************************************************************************
    
    
//    while(stack.size() > 2 || buffer.size() > 1) {
    while(true){
	if(prev_a == 'T') break;
      // get list of possible actions for the current parser state
//if(DEBUG) cerr<< "action_count " << action_count <<"\n";
      current_valid_actions.clear();
//if(DEBUG) cerr<< "unary: " << unary << "nopen_parens: "<<nopen_parens<<"\n";
      for (auto a: possible_actions) {
        if (IsActionForbidden_Discriminative(adict.Convert(a), prev_a, buffer.size(), stack.size(), nopen_parens, unary))
          continue;
        current_valid_actions.push_back(a);
      }
if(DEBUG){
	cerr <<"*************************CONFIGURATION********************** "<<"\n";
	cerr <<"current_valid_actions: "<<current_valid_actions.size()<<" :";
	for(unsigned i = 0; i < current_valid_actions.size(); i ++){
		cerr<<adict.Convert(current_valid_actions[i])<<" ";
	}
	cerr <<"\n";
//	cerr <<"AA1" <<endl;
	unsigned j = 999;
	for(unsigned i = 0; i < current_valid_actions.size(); i ++){
                if(current_valid_actions[i] == correct_actions[action_count]) {j = i; break;}
        }
	if(j == 999){
		cerr<<"gold out\n";
//		exit(1);
	}
//	cerr <<"AA" <<endl;
}
      //cerr << "valid actions = " << current_valid_actions.size() << endl;

      // p_t = pbias + S * slstm + B * blstm + A * almst
      Expression stack_summary = stack_lstm.back();
      Expression action_summary = action_lstm.back();
      Expression buffer_summary = buffer_lstm->back();

      if (apply_dropout) {
        stack_summary = dropout(stack_summary, DROPOUT);
        action_summary = dropout(action_summary, DROPOUT);
        buffer_summary = dropout(buffer_summary, DROPOUT);
      }
      Expression p_t = affine_transform({pbias, S, stack_summary, B, buffer_summary, A, action_summary});
      Expression nlp_t = rectify(p_t);
      //if (build_training_graph) nlp_t = dropout(nlp_t, 0.4);
      // r_t = abias + p2a * nlp
      Expression r_t = affine_transform({abias, p2a, nlp_t});
      if (sample && ALPHA != 1.0f) r_t = r_t * ALPHA;
      // adist = log_softmax(r_t, current_valid_actions)
      Expression adiste = log_softmax(r_t, current_valid_actions);
      vector<float> adist = as_vector(hg->incremental_forward());
      double best_score = adist[current_valid_actions[0]];
      unsigned model_action = current_valid_actions[0];
      if (sample) {
        double p = rand01();
        assert(current_valid_actions.size() > 0);
        unsigned w = 0;
        for (; w < current_valid_actions.size(); ++w) {
          p -= exp(adist[current_valid_actions[w]]);
          if (p < 0.0) { break; }
        }
        if (w == current_valid_actions.size()) w--;
        model_action = current_valid_actions[w];
      } else { // max
        for (unsigned i = 1; i < current_valid_actions.size(); ++i) {
          if (adist[current_valid_actions[i]] > best_score) {
            best_score = adist[current_valid_actions[i]];
            model_action = current_valid_actions[i];
          }
        }
      }
      

      
      unsigned action = model_action;
      unsigned action_to_follow = model_action;
      if (build_training_graph) {  // if we have reference actions (for training) use the reference action
    	  if (action_count >= correct_actions.size()) {
    	          	if(DEBUG)cerr << "Correct action list exhausted, but not in final parser state.\n";
    	            
    	            action = correct_actions[0];
    	          }else{
    	          	action = correct_actions[action_count];
    	          }
      } else {
        //cerr << "Chosen action: " << adict.Convert(action) << endl;
      }
      //cerr << "prob ="; for (unsigned i = 0; i < adist.size(); ++i) { cerr << ' ' << adict.Convert(i) << ':' << adist[i]; }
      
      if (build_training_graph){
           const string& actionGold=adict.Convert(action);
           const string& actionPredicted=adict.Convert(model_action);




         vector<int> costs;
         int higher_cost=0;
     	for (unsigned i = 0; i < current_valid_actions.size(); ++i) {
     		int cost=num_reachable_constituents(adict.Convert(current_valid_actions[i]), pstack, pindex_buffer, gold_constituents, predicted_constituents,gold_init, predicted_init, predicted_unique_init, predicted_arity_init, gold_unique_init,gold_arity_init);
     		costs.push_back(cost);
     		if(cost>higher_cost)
     		{
     			higher_cost=cost;
     		}
     	}
     	
     	
     	double best_optimal_score = -1000;
     	unsigned optimal_action = current_valid_actions[0];
     	
     	
     	double best_nonoptimal_score = -1000;
     	unsigned nonoptimal_action = current_valid_actions[0];
     	int num_optimal=0;
     	bool predicted_is_optimal=false;
     	for (unsigned i = 0; i < current_valid_actions.size(); ++i) {
     		if(costs[i]==higher_cost){
     			num_optimal++;
     		}
     		
     		if(costs[i]==higher_cost && best_optimal_score<adist[current_valid_actions[i]])
     		{
     			best_optimal_score=adist[current_valid_actions[i]];
     			optimal_action=current_valid_actions[i];
     		}
     		
     		
     		if(costs[i]==higher_cost && model_action==current_valid_actions[i]){
     			predicted_is_optimal=true;
     		}
     		
     		if(costs[i]<higher_cost && best_nonoptimal_score<adist[current_valid_actions[i]])
     		{
     					best_nonoptimal_score=adist[current_valid_actions[i]];
     					nonoptimal_action=current_valid_actions[i];
     		}
     	}
     	
     	
     	if(predicted_is_optimal){
     		(*right)++;
     		action=model_action;
     	}
     	else{
     		action=optimal_action;
     		
     	}
     	
     	
     	
     	double p=rand01();
     		if((adist[action]<adist[nonoptimal_action] &&  p>0.9))
     		{	
     			action_to_follow=nonoptimal_action;
     		}else{
     		    	action_to_follow=action;
     		}

     }	

      ++action_count;
      log_probs.push_back(pick(adiste, action));
      results.push_back(action);

      if (build_training_graph) { action = action_to_follow; }
      
      Expression actione = lookup(*hg, p_a, action);
      action_lstm.add_input(actione);

      const string& actionString=adict.Convert(action);
      const char ac = actionString[0];
      const char ac2 = actionString[1];
      if (ac =='S' && ac2=='H') {  // SHIFT
        assert(buffer.size() > 1); // dummy symbol means > 1 (not >= 1)
          stack.push_back(buffer.back());
          stack_lstm.add_input(buffer.back());
          stacki.push_back(bufferi.back());
          buffer.pop_back();
          buffer_lstm->rewind_one_step();
          bufferi.pop_back();
          is_open_paren.push_back(-1);
	unary = 0;
num_trans++;

	if (build_training_graph){
	   int bg=pindex_buffer;
				pindex_buffer++;
				int en=pindex_buffer;
				string elto=std::to_string(bg)+"#"+std::to_string(en);
				pstack.push_back(elto);
	}


      } else if (ac == 'N') { // PJ
        ++nopen_parens;
        assert(stack.size() > 1);
        auto it = action2NTindex.find(action);
        assert(it != action2NTindex.end());
        int nt_index = it->second;
        nt_count++;
        Expression nt_embedding = lookup(*hg, p_nt, nt_index);
        stack.push_back(nt_embedding);
        stack_lstm.add_input(nt_embedding);
        stacki.push_back(-1);
        is_open_paren.push_back(nt_index);
num_trans++;

       	if (build_training_graph){
       		string bg=std::to_string(pindex_buffer);
       		    				const string& NT = actionString.substr(3,actionString.length()-4);
       		    				
       		    				
       		    				string nt=NT+"#"+bg;
       		    				int pos=contain(predicted_unique_init,NT);
       		    				
       		    				string nto=nt;
       		    				     int order=1;
       		    					if(pos!=-1)
       		    			      	   {
       		    						    				   	  predicted_arity_init[pos]=predicted_arity_init[pos]+1;
       		    			 		    				   	      order=predicted_arity_init[pos];
       		    			 		    				   	      nto=nt+"%"+std::to_string(order);
       		    												  bool found_init_in_gold=false;
       		    												  for(unsigned i = 0; i < gold_init.size(); i ++){
       		    															if(nto.compare(gold_init[i])==0){
       		    																	found_init_in_gold=true;
       		    																	break;
       		    															}
       		    												  }
       		    													if(!found_init_in_gold)
       		    													{
       		    														order=111;
       		    													}
       		    			       		    				   }else{
       		    			       		    					   
       		    			       		    					nto=nt+"%"+std::to_string(order);
       		    			       		    					bool found_init_in_gold=false;
       		    			       		    					 for(unsigned i = 0; i < gold_init.size(); i ++){
       		    			       		    											if(nto.compare(gold_init[i])==0){
       		    			       		    													found_init_in_gold=true;
       		    			       		    													break;
       		    			       		    											}
       		    			       		    					 }
       		    			       		    					if(!found_init_in_gold)
       		    			       		    					{
       		    			       		    							order=111;
       		    			     		    					}

       		    			      		    				   	  predicted_unique_init.push_back(NT);
       		    			       		    				   	  predicted_arity_init.push_back(int(1));

       		    			     		    				   }
			    				
			    				nto=nt+"%"+std::to_string(order);
       		    			predicted_init.push_back(nto);
       		    			pstack.push_back("N"+nto);
       	}


      } else if (ac == 'R'){ // REDUCE
num_trans++;
        --nopen_parens;
	if(prev_a == 'N') unary += 1;
	if(prev_a == 'R') unary = 0;
        assert(stack.size() > 2); // dummy symbol means > 2 (not >= 2)
        // find what paren we are closing
        int i = is_open_paren.size() - 1;
        while(is_open_paren[i] < 0) { --i; assert(i >= 0); }
        Expression nonterminal = lookup(*hg, p_ntup, is_open_paren[i]);
        int nchildren = is_open_paren.size() - i - 1;
        assert(nchildren+1 > 0);
        vector<Expression> children(nchildren);
        const_lstm_fwd.start_new_sequence();
        const_lstm_rev.start_new_sequence();

        // REMOVE EVERYTHING FROM THE STACK THAT IS GOING
        // TO BE COMPOSED INTO A TREE EMBEDDING
        for (i = 0; i < nchildren; ++i) {
          children[i] = stack.back();
          stacki.pop_back();
          stack.pop_back();
          stack_lstm.rewind_one_step();
          is_open_paren.pop_back();
        }

        is_open_paren.pop_back(); // nt symbol
        assert (stacki.back() == -1);
        stacki.pop_back(); // nonterminal dummy
        stack.pop_back(); // nonterminal dummy
        stack_lstm.rewind_one_step(); // nt symbol 

	children.push_back(stack.back()); // leftmost
	stacki.pop_back(); //leftmost
	stack.pop_back(); //leftmost
	stack_lstm.rewind_one_step(); //leftmost
	is_open_paren.pop_back(); //leftmost

	nchildren ++;
        // BUILD TREE EMBEDDING USING BIDIR LSTM
        const_lstm_fwd.add_input(nonterminal);
        const_lstm_rev.add_input(nonterminal);
        for (i = 0; i < nchildren; ++i) {
          const_lstm_fwd.add_input(children[i]);
          const_lstm_rev.add_input(children[nchildren - i - 1]);
        }
        Expression cfwd = const_lstm_fwd.back();
        Expression crev = const_lstm_rev.back();
        if (apply_dropout) {
          cfwd = dropout(cfwd, DROPOUT);
          crev = dropout(crev, DROPOUT);
        }
        Expression c = concatenate({cfwd, crev});
        Expression composed = rectify(affine_transform({cbias, cW, c}));
        stack_lstm.add_input(composed);
        stack.push_back(composed);
        stacki.push_back(999); // who knows, should get rid of this
        is_open_paren.push_back(-1); // we just closed a paren at this position
        
        
            	if (build_training_graph){
            		const string& delimiterA = "#";
            		const string& delimiterB = "%";
            		    				string NT;
            		    				string bg;
            		    				string en=std::to_string(pindex_buffer);
            		    				string order="100";
            		    				while(!pstack.empty())
            		    				{
            		    					string elto=pstack.back();
            		    				    pstack.pop_back();
            		    					if(elto[0]=='N')
            		    					{
            		    						string use1 = elto.substr(1);
            		    						string use = use1.substr(0,use1.find(delimiterB));
            		    						order= use1.substr(use1.find(delimiterB)+1);
            		    						NT = use.substr(0,use.find(delimiterA));
				        				string elto2=pstack.back();
            		    						pstack.pop_back();
            		    						bg = elto2.substr(0,elto2.find(delimiterA));
            		    						break;
            		    					}
            		    				}
            		    				pstack.push_back(bg+"#"+en);
            		    				string nt=NT+"|"+bg+"#"+en+"%"+order;
            		    				predicted_constituents.push_back(nt);
            	}
        
        
      }
      else{// TERMINATE
num_trans++;
      }
      prev_a = ac;
      
      
     
    }
    
    assert(stack.size() == 2); // guard symbol, root
    assert(stacki.size() == 2);
    assert(buffer.size() == 1); // guard symbol
    assert(bufferi.size() == 1);
    Expression tot_neglogprob = -sum(log_probs);
    assert(tot_neglogprob.pg != nullptr);
    return results;
  }




struct ParserState {
  LSTMBuilder stack_lstm;
  LSTMBuilder *buffer_lstm;
  LSTMBuilder action_lstm;
  vector<Expression> buffer;
  vector<int> bufferi;
  LSTMBuilder const_lstm_fwd;
  LSTMBuilder const_lstm_rev;

  vector<Expression> stack;
  vector<int> stacki;
  vector<unsigned> results;  // sequence of predicted actions
  bool complete;
  vector<Expression> log_probs;
  double score;
  int action_count;
  int nopen_parens;
  char prev_a;
};


struct ParserStateCompare {
  bool operator()(const ParserState& a, const ParserState& b) const {
    return a.score > b.score;
  }
};


static int contain(vector<string>& vector, string a)
{
	for(unsigned i = 0; i < vector.size(); i ++){
		if(vector[i].compare(a)==0)return i;
	}
	return -1;
}

static bool is_sort(string span1, string span2, vector<string>& gold_inits)
{
	const string& delimiterA = "#";
	const string& delimiterB = "%";
	
	string use = span1.substr(1);
	string use1 = use.substr(0,use.find(delimiterB));
			string order1 = use.substr(use.find(delimiterB)+1);
			string NT1 = use1.substr(0,use1.find(delimiterA));
			string l1 = use1.substr(use1.find(delimiterA)+1);
	
				
			string use0 = span2.substr(1);
			string use2 = use0.substr(0,use0.find(delimiterB));
									string order2 = use0.substr(use0.find(delimiterB)+1);
						string NT2 = use2.substr(0,use2.find(delimiterA));
						string l2 = use2.substr(use2.find(delimiterA)+1);		

	if(NT1.compare(NT2)==0 && order1.compare(order2)==0) return true;
					

		
						int pos1=-1;
						int pos2=-1;
						
						for(unsigned i = 0; i < gold_inits.size(); i ++){
							string use10 = gold_inits[i];
							string use11 = use10.substr(0,use10.find(delimiterB));
							string order = use10.substr(use10.find(delimiterB)+1);
							string NT = use11.substr(0,use11.find(delimiterA));
							string l = use.substr(use.find(delimiterA)+1);
						
							if(NT.compare(NT1)==0 && order.compare(order1)==0  && pos1==-1)
							{
								pos1=i;
							}
							
							if(NT.compare(NT2)==0 && order.compare(order2)==0 && pos2==-1) 
									{
										pos2=i;
									}
							
							if(pos1!=-1 && pos2!=-1)break;
						}
						
					
						if(pos1==-1 || pos2==-1){
							return false;
						}
						
						
						
						 return pos1<pos2;
		
	
}


static int num_reachable_constituents(string tran, vector<string> stack, int index_buffer, vector<string>& gold_constituents, vector<string> predicted_constituents, vector<string>& gold_init, vector<string> predicted_init, vector<string> predicted_unique_init, vector<int> predicted_arity_init, vector<string>& gold_unique_init, vector<int>& gold_arity_init) {
	const string& delimiterA = "|";
	const string& delimiterB = "#";
	const string& delimiterD = "%";
	
	bool DEBUG2=true;
	
	
	if(stack.size()==0)return gold_constituents.size();
	
	
	
	int label_penalty=0;
	int nt_penalty=0;
	
	if(tran[0] == 'S'){
		int bg=index_buffer;
		index_buffer++;
		int en=index_buffer;
		string elto=std::to_string(bg)+"#"+std::to_string(en);
		stack.push_back(elto);
	
	}else if(tran[0] == 'N'){
		
								string bg=std::to_string(index_buffer);
		       		    				const string& NT = tran.substr(3,tran.length()-4);
		       		    				
		       		    				string nt=NT+"#"+bg;
		       		    				int pos=contain(predicted_unique_init,NT);
		       		    				string nto=nt;
	       		    				 int order=1;
				    				if(pos!=-1)
      		    				   {
				    				  
       		    				   	  predicted_arity_init[pos]=predicted_arity_init[pos]+1;
 		    				   	      order=predicted_arity_init[pos];
 		    				   	      nto=nt+"%"+std::to_string(order);
									  bool found_init_in_gold=false;
									  for(unsigned i = 0; i < gold_init.size(); i ++){
												if(nto.compare(gold_init[i])==0){
														found_init_in_gold=true;
														break;
												}
									  }
										if(!found_init_in_gold)
										{
													nt_penalty=1;
																													order=111;
										}
       		    				   }else{
       		    					   
       		    					nto=nt+"%"+std::to_string(order);
       		    					bool found_init_in_gold=false;
       		    					 for(unsigned i = 0; i < gold_init.size(); i ++){
       		    											if(nto.compare(gold_init[i])==0){
       		    													found_init_in_gold=true;
       		    													break;
       		    											}
       		    					 }
       		    					if(!found_init_in_gold)
       		    					{
       		    							nt_penalty=1;
       		    							order=111;
     		    					}
     		    					
      		    				   	  predicted_unique_init.push_back(NT);
       		    				   	  predicted_arity_init.push_back(int(1));

     		    				   }
     		    				   
       		    				
       		    				
       		    				
       		    				nto=nt+"%"+std::to_string(order);	   
       		    				     		    				   
       		    			  predicted_init.push_back(nto);
       		    			  stack.push_back("N"+nto);
					
	
	}else if(tran[0] == 'R')
				{
				
		        		    				string NT;
		        		    				string bg;
		        		    				string en=std::to_string(index_buffer);
		        		    				string order="100";
		        		    				while(!stack.empty())
		        		    				{
		        		    					string elto=stack.back();
		        		    					stack.pop_back();
		        		    					if(elto[0]=='N')
		        		    					{
		        		    						string use1 = elto.substr(1);
		        		    						string use = use1.substr(0,use1.find(delimiterD));
		        		    						order= use1.substr(use1.find(delimiterD)+1);
		        		    						NT = use.substr(0,use.find(delimiterB));
		        		    						string elto2=stack.back();
		        		    						stack.pop_back();
		        		    						bg = elto2.substr(0,elto2.find(delimiterB));
		        		    						break;
		        		    					}
		        		    					
		        		    				}
		        		    				
		        		    				stack.push_back(bg+"#"+en);
		        		    				string nt=NT+"|"+bg+"#"+en+"%"+order;
		        		    				predicted_constituents.push_back(nt);
				
					
		        		    				bool found_in_gold=false;
		        		    				int num_veces_in_gold=0;
		        		    				for(unsigned i = 0; i < gold_constituents.size(); i ++){
		        		    							if(nt.compare(gold_constituents[i])==0){
		        		    								found_in_gold=true;
		        		    								num_veces_in_gold++;
		        		    							   
		        		    							}
		        		    																	
		        		    				}
		        		    				if(!found_in_gold){
		        		    					label_penalty=1;
		        		    				
		        		    				}else if(found_in_gold){
		        		    						int num_veces_creado=0;
		        		    				        for(unsigned i = 0; i < predicted_constituents.size(); i ++){
		        		    								if(nt.compare(predicted_constituents[i])==0){
		        		    											num_veces_creado++;
		        		    							  			    if(num_veces_creado>num_veces_in_gold)
		        		    											{
		        		    													label_penalty=1;
		        		    													break;
		        		    											}
		        		    							    }
		        		    						}
		        		    				}
					
	}
	
	

	

	 vector<string> reachable;
	 string last_span=stack.back();
	       
	 
	 string span_k;
	 string span_j;
	 
	 if(last_span[0]!='N')
	 {
	    span_k=last_span.substr(0,last_span.find(delimiterB));
		span_j=last_span.substr(last_span.find(delimiterB)+1);
		
	 }else{
		 
		 string use1 = last_span.substr(1);
		 string use = use1.substr(0,use1.find(delimiterD));
		 string kkorder= use1.substr(use1.find(delimiterD)+1);
 		 string uNT = use.substr(0,use.find(delimiterB));
 		 span_k = use.substr(use.find(delimiterB)+1);
 		 span_j=span_k;
	 }
		
		for(unsigned i = 0; i < gold_constituents.size(); i ++){

			bool already_created=false;
			for(unsigned elto = 0; elto < predicted_constituents.size(); elto ++)
						{
							if(predicted_constituents[elto].compare(gold_constituents[i])==0)
							{
								already_created=true;		
								break;
							}
						}
	        if(already_created)
			{		
				reachable.push_back(gold_constituents[i]);
				continue;
			}
	       
	  
	        if(tran[0] != 'T'){
				
	        	string constituent=gold_constituents[i].substr(0,gold_constituents[i].find(delimiterD));
	        					        				string order=gold_constituents[i].substr(gold_constituents[i].find(delimiterD)+1);
	        					        					        		        
	        					        				string NT=constituent.substr(0,constituent.find(delimiterA));
	        					        				string span=constituent.substr(constituent.find(delimiterA)+1);
	        					        								
	        					        				string m=span.substr(0,span.find(delimiterB));
	        					        				string n=span.substr(span.find(delimiterB)+1);
				
				
				if( atoi(span_j.c_str())<=atoi(m.c_str()) && atoi(m.c_str())<atoi(n.c_str()) )
				{
					reachable.push_back(gold_constituents[i]);	
					continue;
				}
				
	
				bool wrong_span_init=true;
				bool found_NT=false;
				for(int elto = stack.size()-1; elto >=0; elto--){
					string cad=stack[elto];
					if(cad[0]=='N')
				    {
						string use1 = cad.substr(1);
						string use = use1.substr(0,use1.find(delimiterD));
						string porder=use1.substr(use1.find(delimiterD)+1);
						string pNT = use.substr(0,use.find(delimiterB));
						if(pNT.compare(NT)==0 && porder.compare(order)==0)
						{
							if(elto-1>=0)
							{
								string before=stack[elto-1];
								string bg=before.substr(0,before.find(delimiterB));
								string en=before.substr(before.find(delimiterB)+1);
								if(atoi(m.c_str())==atoi(bg.c_str()) && atoi(m.c_str())!=atoi(span_j.c_str()) && atoi(span_j.c_str())<=atoi(n.c_str()))
								{	
									bool is_sort_b=true;
									for(unsigned elto2 = elto+1; elto2 < stack.size(); elto2 ++){
									    string cad2=stack[elto2];
										if(cad2[0]=='N')
										{
											if(!is_sort("N"+NT+"#1000%"+order,cad2,gold_init))
										    {
												is_sort_b=false;
														break;
											}
										}
									}
									for(unsigned elto2 = 0; elto2 < elto+1; elto2 ++){
										string cad2=stack[elto2];
										if(cad2[0]=='N')
										{
											if(!is_sort(cad2,"N"+NT+"#1000%"+order,gold_init))
										    {
												is_sort_b=false;
												break;
											}
										}
									}
									
									if(is_sort_b)wrong_span_init=false;
								}
							}
							found_NT=true;
							break;
						}
					}
				}

				
				if(!found_NT)
				{
					for(int elto = stack.size()-1; elto >=0; elto--){
							string cad=stack[elto];
							if(cad[0]!='N')
							{
								string bg = cad.substr(0,cad.find(delimiterB));
								string en=cad.substr(cad.find(delimiterB)+1);
								if(atoi(m.c_str())==atoi(bg.c_str())
										&& atoi(m.c_str())!=atoi(span_j.c_str()) && atoi(span_j.c_str())<=atoi(n.c_str()))
								{	
									
									if(elto+1<stack.size())
									{
										string after=stack[elto+1];
										if(after[0]=='N')
										{
											bool is_sort_b=true;
																				
											
											for(unsigned elto2 = 0; elto2 < stack.size(); elto2 ++){
												    string cad2=stack[elto2];
													if(cad2[0]=='N')
													{
															if(!is_sort(cad2,"N"+NT+"#1000%"+order,gold_init))
														    {
																	is_sort_b=false;
																	break;
															}
													}
											}
											if(is_sort_b)wrong_span_init=false;
											
											
										}
									}else if(elto+1==stack.size()){
										wrong_span_init=false;
									}
									
									break;
								}
							}
					}
				}
				
				
				
				if(!already_created && !wrong_span_init)
				{
					
					reachable.push_back(gold_constituents[i]);	
					
				}
	        }
		}

		
		return reachable.size()-label_penalty-nt_penalty;
	
}



static void prune(vector<ParserState>& pq, unsigned k) {
  if (pq.size() == 1) return;
  if (k > pq.size()) k = pq.size();
  partial_sort(pq.begin(), pq.begin() + k, pq.end(), ParserStateCompare());
  pq.resize(k);
  reverse(pq.begin(), pq.end());
  //cerr << "PRUNE\n";
  //for (unsigned i = 0; i < pq.size(); ++i) {
  //  cerr << pq[i].score << endl;
  //}
}

static bool all_complete(const vector<ParserState>& pq) {
  for (auto& ps : pq) if (!ps.complete) return false;
  return true;
}


};
void signal_callback_handler(int /* signum */) {
  if (requested_stop) {
    cerr << "\nReceived SIGINT again, quitting.\n";
    _exit(1);
  }
  cerr << "\nReceived SIGINT terminating optimization early...\n";
  requested_stop = true;
}

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv, 1989121011);

  cerr << "COMMAND LINE:"; 
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
  cerr << endl;
  unsigned status_every_i_iterations = 100;

  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);
  DEBUG = conf.count("debug");
  USE_POS = conf.count("use_pos_tags");
  if (conf.count("dropout"))
    DROPOUT = conf["dropout"].as<float>();
  LAYERS = conf["layers"].as<unsigned>();
  INPUT_DIM = conf["input_dim"].as<unsigned>();
  PRETRAINED_DIM = conf["pretrained_dim"].as<unsigned>();
  HIDDEN_DIM = conf["hidden_dim"].as<unsigned>();
  ACTION_DIM = conf["action_dim"].as<unsigned>();
  LSTM_INPUT_DIM = conf["lstm_input_dim"].as<unsigned>();
  POS_DIM = conf["pos_dim"].as<unsigned>();
  if (conf.count("train") && conf.count("dev_data") == 0) {
    cerr << "You specified --train but did not specify --dev_data FILE\n";
    return 1;
  }
  if (conf.count("alpha")) {
    ALPHA = conf["alpha"].as<float>();
    if (ALPHA <= 0.f) { cerr << "--alpha must be between 0 and +infty\n"; abort(); }
  }
  if (conf.count("samples")) {
    N_SAMPLES = conf["samples"].as<unsigned>();
    if (N_SAMPLES == 0) { cerr << "Please specify N>0 samples\n"; abort(); }
  }
  
  ostringstream os;
  os << "ntparse"
     << (USE_POS ? "_pos" : "")
     << '_' << IMPLICIT_REDUCE_AFTER_SHIFT
     << '_' << LAYERS
     << '_' << INPUT_DIM
     << '_' << HIDDEN_DIM
     << '_' << ACTION_DIM
     << '_' << LSTM_INPUT_DIM
     << "-pid" << getpid() << ".params";
  const string fname = os.str();
  cerr << "PARAMETER FILE: " << fname << endl;
  bool softlinkCreated = false;

  Model model;

  parser::KOracle corpus(&termdict, &adict, &posdict, &ntermdict);
  parser::KOracle dev_corpus(&termdict, &adict, &posdict, &ntermdict);
  parser::KOracle test_corpus(&termdict, &adict, &posdict, &ntermdict);
  corpus.load_oracle(conf["training_data"].as<string>(), true);	
  corpus.load_bdata(conf["bracketing_dev_data"].as<string>());

  if (conf.count("words"))
    parser::ReadEmbeddings_word2vec(conf["words"].as<string>(), &termdict, &pretrained);

  // freeze dictionaries so we don't accidentaly load OOVs
  termdict.Freeze();
  termdict.SetUnk("UNK"); // we don't actually expect to use this often
     // since the Oracles are required to be "pre-UNKified", but this prevents
     // problems with UNKifying the lowercased data which needs to be loaded
  adict.Freeze();
  ntermdict.Freeze();
  posdict.Freeze();

  /*ofstream ofs;
  ofs.open("terminal.dict");
  for(int i = 0; i < termdict.size(); ++i) ofs << termdict.Convert(i)<<"\n";
  ofs.close();

  ofs.open("non-terminal.dict");
  for(int i = 0; i < ntermdict.size(); ++i) ofs << ntermdict.Convert(i)<<"\n";
  ofs.close();

  ofs.open("action..dict");
  for(int i = 0; i < adict.size(); ++i) ofs << adict.Convert(i)<<"\n";
  ofs.close();

  ofs.open("pos.dict");
  for(int i = 0; i < posdict.size(); ++i) ofs << posdict.Convert(i)<<"\n";
  ofs.close();

  exit(1); */
  {  // compute the singletons in the parser's training data
    unordered_map<unsigned, unsigned> counts;
    for (auto& sent : corpus.sents)
      for (auto word : sent.raw) counts[word]++;
    singletons.resize(termdict.size(), false);
    for (auto wc : counts)
      if (wc.second == 1) singletons[wc.first] = true;
  }

  if (conf.count("dev_data")) {
    cerr << "Loading validation set\n";
    dev_corpus.load_oracle(conf["dev_data"].as<string>(), false);
  }
  if (conf.count("test_data")) {
    cerr << "Loading test set\n";
    test_corpus.load_oracle(conf["test_data"].as<string>(), false);
  }

  for (unsigned i = 0; i < adict.size(); ++i) {
    const string& a = adict.Convert(i);
    if (a[0] != 'N') continue;
    size_t start = a.find('(') + 1;
    size_t end = a.rfind(')');
    int nt = ntermdict.Convert(a.substr(start, end - start));
    action2NTindex[i] = nt;
  }

  NT_SIZE = ntermdict.size();
  POS_SIZE = posdict.size();
  VOCAB_SIZE = termdict.size();
  ACTION_SIZE = adict.size();
  possible_actions.resize(adict.size());
  for (unsigned i = 0; i < adict.size(); ++i)
    possible_actions[i] = i;

  ParserBuilder parser(&model, pretrained);
  if (conf.count("model")) {
    ifstream in(conf["model"].as<string>().c_str());
    boost::archive::text_iarchive ia(in);
    ia >> model;
  }

  //TRAINING
  if (conf.count("train")) {
    signal(SIGINT, signal_callback_handler);
    SimpleSGDTrainer sgd(&model);
    //AdamTrainer sgd(&model);
    //MomentumSGDTrainer sgd(&model);
    //sgd.eta_decay = 0.08;
    sgd.eta_decay = 0.05;
    vector<unsigned> order(corpus.sents.size());
    for (unsigned i = 0; i < corpus.sents.size(); ++i)
      order[i] = i;
    double tot_seen = 0;
    status_every_i_iterations = min((int)status_every_i_iterations, (int)corpus.sents.size());
    unsigned si = corpus.sents.size();
    cerr << "NUMBER OF TRAINING SENTENCES: " << corpus.sents.size() << endl;
    unsigned trs = 0;
    unsigned words = 0;
    double right = 0;
    double llh = 0;
    bool first = true;
    int iter = -1;
    double best_dev_err = 9e99;
    double bestf1=0.0;
    //cerr << "TRAINING STARTED AT: " << put_time(localtime(&time_start), "%c %Z") << endl;
    while(!requested_stop) {
      //if(iter == 2) exit(0);
      ++iter;
      CUR_ITER = iter;
      auto time_start = chrono::system_clock::now();
      for (unsigned sii = 0; sii < status_every_i_iterations; ++sii) {
           if (si == corpus.sents.size()) {
             si = 0;
             if (first) { first = false; } else { sgd.update_epoch(); }
             cerr << "**SHUFFLE\n";
             cerr << "**NEW INORDER DYNAMIC ORACLE\n";
             random_shuffle(order.begin(), order.end());
           }
           tot_seen += 1;
           auto& sentence = corpus.sents[order[si]];
	   const vector<int>& actions=corpus.actions[order[si]];
           ComputationGraph hg;
           parser.log_prob_parser(&hg,sentence,actions,&right,false);
           double lp = as_scalar(hg.incremental_forward());
           if (lp < 0) {
             cerr << "Log prob < 0 on sentence " << order[si] << ": lp=" << lp << endl;
             assert(lp >= 0.0);
           }
           hg.backward();
           sgd.update(1.0);
           llh += lp;
           ++si;
           trs += actions.size();
           words += sentence.size();
      }
      sgd.status();
      auto time_now = chrono::system_clock::now();
      auto dur = chrono::duration_cast<chrono::milliseconds>(time_now - time_start);
      cerr << "update #" << iter << " (epoch " << (tot_seen / corpus.sents.size()) <<
         /*" |time=" << put_time(localtime(&time_now), "%c %Z") << ")\tllh: "<< */
        ") per-action-ppl: " << exp(llh / trs) << " per-input-ppl: " << exp(llh / words) << " per-sent-ppl: " << exp(llh / status_every_i_iterations) << " err: " << (trs - right) / trs << " [" << dur.count() / (double)status_every_i_iterations << "ms per instance]" << endl;
      llh = trs = right = words = 0;

      static int logc = 0;
      ++logc;
      
      	if((tot_seen / corpus.size())>70.0)break;
      
      if (logc % 25 == 1) { // report on dev set
	/*std::string command_a="./aa" ;
	const char* cmd_a=command_a.c_str();
	cerr<<cmd_a<<"\n";
	cerr<<system(cmd_a)<<"\n";
	cerr<<"NO FUCA"<<"\n";
	cerr<<execl ("./aa", "aa", (char*) NULL)<<"\n";*/



        unsigned dev_size = dev_corpus.size();
        double llh = 0;
        double trs = 0;
        double right = 0;
        double dwords = 0;
        ofstream out("dev.act");
        auto t_start = chrono::high_resolution_clock::now();
        for (unsigned sii = 0; sii < dev_size; ++sii) {
           const auto& sentence=dev_corpus.sents[sii];
	   const vector<int>& actions=dev_corpus.actions[sii];
           dwords += sentence.size();
           {  ComputationGraph hg;
              parser.log_prob_parser(&hg,sentence,actions,&right,true);
              double lp = as_scalar(hg.incremental_forward());
              llh += lp;
           }
           ComputationGraph hg;
           vector<unsigned> pred = parser.log_prob_parser(&hg,sentence,vector<int>(),&right,true);
           unsigned ti = 0;
           for (auto a : pred) {
                out << adict.Convert(a);
                if(adict.Convert(a) == "SHIFT"){
                        out<<" " << posdict.Convert(sentence.pos[ti])<< " " <<sentence.surfaces[ti];
                        ti++;
                }
                out<<endl;
           }
           out << endl;
           double lp = 0;
           trs += actions.size();
        }
        auto t_end = chrono::high_resolution_clock::now();
        out.close();
        double err = (trs - right) / trs;

	std::string command_1="python mid2tree.py dev.act > dev.eval" ;
	const char* cmd_1=command_1.c_str();
	cerr<<system(cmd_1)<<"\n";


        std::string command2="EVALB/evalb -p EVALB/COLLINS.prm "+corpus.devdata+" dev.eval > dev.evalout";
        const char* cmd2=command2.c_str();

        system(cmd2);
        
        std::ifstream evalfile("dev.evalout");
        std::string lineS;
        std::string brackstr="Bracketing FMeasure";
        double newfmeasure=0.0;
        std::string strfmeasure="";
        bool found=0;
        while (getline(evalfile, lineS) && !newfmeasure){
		if (lineS.compare(0, brackstr.length(), brackstr) == 0) {
			//std::cout<<lineS<<"\n";
			strfmeasure=lineS.substr(lineS.size()-5, lineS.size());
                        std::string::size_type sz;     // alias of size_t

		        newfmeasure = std::stod (strfmeasure,&sz);
			//std::cout<<strfmeasure<<"\n";
		}
        }
        
 
        
        cerr << "  **dev (iter=" << iter << " epoch=" << (tot_seen / corpus.size()) << ")\tllh=" << llh << " ppl: " << exp(llh / dwords) << " f1: " << newfmeasure << " err: " << err << "\t[" << dev_size << " sents in " << chrono::duration<double, milli>(t_end-t_start).count() << " ms]" << endl;
//        if (err < best_dev_err && (tot_seen / corpus.size()) > 1.0) {
       if (newfmeasure>bestf1) {
          cerr << "  new best...writing model to " << fname << " ...\n";
          best_dev_err = err;
	  bestf1=newfmeasure;
	  ostringstream part_os;
  	  part_os << "ntparse"
     	      << (USE_POS ? "_pos" : "")
              << '_' << IMPLICIT_REDUCE_AFTER_SHIFT
              << '_' << LAYERS
              << '_' << INPUT_DIM
              << '_' << HIDDEN_DIM
              << '_' << ACTION_DIM
              << '_' << LSTM_INPUT_DIM
              << "-pid" << getpid() 
	      << "-part" << (tot_seen/corpus.size()) << ".params";
 	  
	  const string part = part_os.str();
 
          ofstream out("model/"+part);
          boost::archive::text_oarchive oa(out);
          oa << model;
          //system("cp dev.eval dev.eval.best");


	  std::string bestmodel = boost::lexical_cast<std::string>((tot_seen/corpus.size()));
  	  std::string command_cp1="cp dev.eval dev.eval.best"+bestmodel ;
	  const char* cmd_cp1=command_cp1.c_str();
          system(cmd_cp1);
	  std::string command_cp2="cp dev.evalout dev.evalout.best"+bestmodel ;
	  const char* cmd_cp2=command_cp2.c_str();
	  system(cmd_cp2);



          // Create a soft link to the most recent model in order to make it
          // easier to refer to it in a shell script.
          /*if (!softlinkCreated) {
            string softlink = " latest_model";
            if (system((string("rm -f ") + softlink).c_str()) == 0 && 
                system((string("ln -s ") + fname + softlink).c_str()) == 0) {
              cerr << "Created " << softlink << " as a soft link to " << fname 
                   << " for convenience." << endl;
            }
            softlinkCreated = true;
          }*/
        }
      }
    }
   
/*	cerr << "  writing last iteration model to " << fname << " ...\n";
          ostringstream part_os;
  	  part_os << "ntparse"
     	      << (USE_POS ? "_pos" : "")
              << '_' << IMPLICIT_REDUCE_AFTER_SHIFT
              << '_' << LAYERS
              << '_' << INPUT_DIM
              << '_' << HIDDEN_DIM
              << '_' << ACTION_DIM
              << '_' << LSTM_INPUT_DIM
              << "-pid" << getpid() 
	      << "-part" << (tot_seen/corpus.size()) << ".params";
 	  
	  const string part = part_os.str();
 
          ofstream out("model/"+part);
          boost::archive::text_oarchive oa(out);
          oa << model;*/


  } // should do training?
  if (test_corpus.size() > 0) { // do test evaluation
        unsigned test_size = test_corpus.size();
        double llh = 0;
        double trs = 0;
        double right = 0;
        double dwords = 0;
        auto t_start = chrono::high_resolution_clock::now();
	const vector<int> actions;
	if(conf.count("samples")>0){
        for (unsigned sii = 0; sii < test_size; ++sii) {
           const auto& sentence=test_corpus.sents[sii];
           dwords += sentence.size();
           for (unsigned z = 0; z < N_SAMPLES; ++z) {
             ComputationGraph hg;
             vector<unsigned> pred = parser.log_prob_parser(&hg,sentence,actions,&right,true,true);
             int ti = 0;
             for (auto a : pred) {
             	cout << adict.Convert(a);
		if (adict.Convert(a) == "SHIFT"){
			cout<<" "<<posdict.Convert(sentence.pos[ti])<< " " <<sentence.surfaces[ti];
			ti++;
		}
		cout << endl;
	     }
             cout << endl;
           }
         }
         }
        ofstream out("test.act");
        t_start = chrono::high_resolution_clock::now();
        for (unsigned sii = 0; sii < test_size; ++sii) {
           const auto& sentence=test_corpus.sents[sii];
           const vector<int>& actions=test_corpus.actions[sii];
/*	   for(unsigned i = 0; i < sentence.size(); i ++){
	   	out << termdict.Convert(sentence.raw[i])<<" ";
	   }
	   out<<"||| ";
           for(unsigned i = 0; i < sentence.size(); i ++){
                out << termdict.Convert(sentence.lc[i])<<" ";
           }
	   out<<"||| ";
	   for(unsigned i = 0; i < sentence.size(); i ++){
                out << posdict.Convert(sentence.pos[i])<<" ";
           }
	   out<<"\n";*/
           dwords += sentence.size();
           {  ComputationGraph hg;
              parser.log_prob_parser(&hg,sentence,actions,&right,true);
              double lp = as_scalar(hg.incremental_forward());
              llh += lp;
           }
           ComputationGraph hg;
           vector<unsigned> pred = parser.log_prob_parser(&hg,sentence,vector<int>(),&right,true);
           unsigned ti = 0;
           for (auto a : pred) {
           	out << adict.Convert(a);
		if(adict.Convert(a) == "SHIFT"){
			out<<" " << posdict.Convert(sentence.pos[ti])<< " " <<sentence.surfaces[ti];
			ti++;
		}
		out<<endl;
	   }
           out << endl;
           double lp = 0;
           trs += actions.size();
        }
        
        auto t_end = chrono::high_resolution_clock::now();
        out.close();
        double err = (trs - right) / trs;

	cerr << "Parsed in " << chrono::duration<double, milli>(t_end-t_start).count() << " ms con "<<num_trans<< " trans" << endl;
	
        std::string command_1="python mid2tree.py test.act > test.eval" ;
        const char* cmd_1=command_1.c_str();
        system(cmd_1);

        std::string command2="EVALB/evalb -p EVALB/COLLINS.prm "+corpus.devdata+" test.eval > test.evalout";
        const char* cmd2=command2.c_str();

        system(cmd2);

        std::ifstream evalfile("test.evalout");
        std::string lineS;
        std::string brackstr="Bracketing FMeasure";
        double newfmeasure=0.0;
        std::string strfmeasure="";
        bool found=0;
        while (getline(evalfile, lineS) && !newfmeasure){
                if (lineS.compare(0, brackstr.length(), brackstr) == 0) {
                        //std::cout<<lineS<<"\n";
                        strfmeasure=lineS.substr(lineS.size()-5, lineS.size());
                        std::string::size_type sz;
                        newfmeasure = std::stod (strfmeasure,&sz);
                        //std::cout<<strfmeasure<<"\n";
                }
        }

       cerr<<"F1score: "<<newfmeasure<<"\n";
  }
}
