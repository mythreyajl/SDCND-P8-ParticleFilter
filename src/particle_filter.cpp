/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <map>

#include "particle_filter.h"

using namespace std;

static double normal_pdf(double mean, double sigma, double x) {
    return (1/(sqrt(2*M_PI)*sigma))*exp(-pow(x-mean,2)/(2*sigma*sigma));
}

static double bivariate_norm_pdf(double mx, double my, double sx, double sy, double x, double y) {
    return normal_pdf(mx, sx, x)*normal_pdf(my, sy, y);
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    
    // Set number of particles
    num_particles = 100;  // TODO: Set the number of particles
    
    // Initialize all particles based on x, y, theta estimate
    for(size_t p = 0; p<num_particles; p++) {
        
        Particle particle = Particle();
        particle.id = int(p);
        particle.x = x;
        particle.y = y;
        particle.t = theta;
        particle.weight = 1.0;
        particles.push_back(particle);
    }
    
    // Random Engine initialization
    default_random_engine generator;
    auto& std_x = std[0];
    auto& std_y = std[1];
    auto& std_t = std[2];
    std::normal_distribution<double> dist_x(0, std_x);
    std::normal_distribution<double> dist_y(0, std_y);
    std::normal_distribution<double> dist_t(0, std_t);
    
    // Add Gaussian noise based on GPS uncertainties
    for(auto& p : particles) {
        
        p.x += dist_x(generator);
        p.y += dist_y(generator);
        p.t += dist_t(generator);
    }
    
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    
    // Setting up Random Engine
    default_random_engine generator;
    auto& std_x = std_pos[0];
    auto& std_y = std_pos[1];
    auto& std_t = std_pos[2];
    std::normal_distribution<double> dist_x(0, std_x);
    std::normal_distribution<double> dist_y(0, std_y);
    std::normal_distribution<double> dist_t(0, std_t);
    
    // Predicting particles
    for(auto& p : particles) {
        
        double dx, dy, dt = 0;
        if(yaw_rate == 0) {
            dx = velocity * cos(p.t) * delta_t;
            dy = velocity * sin(p.t) * delta_t;
        } else {
            dt = yaw_rate * delta_t;
            dx = (velocity/yaw_rate) * (sin(p.t+dt) - sin(p.t));
            dy = (velocity/yaw_rate) * (cos(p.t) - cos(p.t+dt));
        }
        p.x += dx + dist_x(generator);
        p.y += dy + dist_y(generator);
        p.t += dt + dist_t(generator);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    
    // Nearest neighbor based map landmarks with threshold
    for(auto& obs : observations) {
        
        double min_dist = __DBL_MAX__;
        int id;
        for(auto& pred : predicted) {
            
            double dist = sqrt(pow(pred.x - obs.x,2) + pow(pred.y - obs.y,2));
            if(dist < min_dist) {
                id = pred.id;
                min_dist = dist;
            }
        }
        obs.id = id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    
    // Aliases for landmark stds
    double& std_x = std_landmark[0];
    double& std_y = std_landmark[1];
    
    for(auto& p : particles) {
        
        // Prune landmarks based on proximity to particle location
        std::vector<LandmarkObs> pruned_lm_list;
        std::map<int, LandmarkObs> pruned_lm_map;
        for(auto& lm : map_landmarks.landmark_list) {
            // Eliminate landmarks based on sensor range (equivalent of camera frustum)
            
            double dx = lm.x_f - p.x;
            double dy = lm.y_f - p.y;
            double dist = sqrt(dx*dx + dy*dy);
            
            if(dist < sensor_range) {
                LandmarkObs landmark;
                landmark.x = lm.x_f;
                landmark.y = lm.y_f;
                landmark.id = lm.id_i;
                
                pruned_lm_list.push_back(landmark);
                pruned_lm_map[lm.id_i] = landmark;
            }
        }
        
        // Transform observations to map frome
        std::vector<LandmarkObs> observations_map_frame;
        for(auto& obs : observations) {
            double tx = obs.x * cos(p.t) - obs.y * sin(p.t) + p.x;
            double ty = obs.x * sin(p.t) + obs.y * cos(p.t) + p.y;
            LandmarkObs tf_obs;
            tf_obs.id = -1;
            tf_obs.x = tx;
            tf_obs.y = ty;
            observations_map_frame.push_back(tf_obs);
        }
        dataAssociation(pruned_lm_list, observations_map_frame);
        
        // Update weights before importance resampling
        double w = 1;
        for(auto& ao : observations_map_frame) {
            auto lm = pruned_lm_map[ao.id];
            w *= bivariate_norm_pdf(lm.x, lm.y, std_x, std_y, ao.x, ao.y);
        }
        p.weight = w;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    // Sum total of weights
    double total_weight = 0.0;
    for(auto& p : particles) {
        total_weight += p.weight;
    }
    
    // Normalized weights
    for(auto& p : particles) {
        p.weight /= total_weight;
    }
    
    // Generate cumulative weights
    std::vector<double> cumulative_weights;
    double prev = 0.0;
    for(auto& p : particles) {
        double current_cw = prev + p.weight;
        cumulative_weights.push_back(current_cw);
        prev = current_cw;
    }
    cumulative_weights.back() = 1.0;
    
    // Resample
    std::vector<Particle> new_particles;
    default_random_engine generator;
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    for(size_t i = 0; i < num_particles; ++i) {
        double random_num = uni(generator);
        int count = 0;
        for(auto cw : cumulative_weights) {
            if(cw > random_num) {
                auto& sample = particles[count];
                sample.weight = 1.0;
                new_particles.push_back(sample);
                break;
            }
            count++;
        }
        auto& sample = particles[num_particles];
        sample.weight = 1.0;
        new_particles.push_back(sample);
        
    }
    std::swap(new_particles, particles);
    
}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
