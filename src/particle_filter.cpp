/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 * Modified on: Apr 19, 2020
 * Author: Artem Kaliuk
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

  num_particles = 100;

  std::random_device rd;
  std::default_random_engine gen(rd());
  // Create a normal distribution with the specified parameters
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  // Create an empty vector of a Particle class with num_particles elements
  particles.resize(num_particles);
  weights.resize(num_particles);
  // Loop over the particles and set up their properties: particle id, x, y coordinates, orientation and weight
  for (int i = 0; i < num_particles; i++){
	  particles[i].id = i;
	  particles[i].x = dist_x(gen);
	  particles[i].y = dist_y(gen);
	  particles[i].theta = dist_theta(gen);
	  particles[i].weight = 1.0f;
	  weights[i] = particles[i].weight;
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {

	std::default_random_engine gen;
	
	// Create a normal distribution for the measurements with specified parameters
	std::normal_distribution<double> dist_x(0.0, std_pos[0]);
	std::normal_distribution<double> dist_y(0.0, std_pos[1]);
	std::normal_distribution<double> dist_theta(0.0, std_pos[2]);
	
	for (int i = 0; i < num_particles; i++){
		if (abs(yaw_rate) > 0.000001){
			// add measurements
			particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
			particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
			particles[i].theta += yaw_rate * delta_t;
		}
		else{
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}
		// add random Gaussian noise to the position and orientation
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations,
									 vector<int>& associations,
									 vector<double>& sense_x,
									 vector<double>& sense_y,
									 vector<double>& assoc_x,
									 vector<double>& assoc_y){

	for (int i = 0; i < observations.size(); i++){
		vector<double> dist_obs2pred (predicted.size());
		for (int j = 0; j < predicted.size(); j++){
			dist_obs2pred[j] = sqrt(pow(observations[i].x - predicted[j].x, 2) + pow(observations[i].y - predicted[j].y, 2));
		}
		// find the index of the closest distance
		int min_dist_idx = std::min_element(dist_obs2pred.begin(), dist_obs2pred.end()) - dist_obs2pred.begin();

		observations[i].id = predicted[min_dist_idx].id;
		associations.push_back(observations[i].id);
		sense_x.push_back(observations[i].x);
		sense_y.push_back(observations[i].y);
		assoc_x.push_back(predicted[min_dist_idx].x);
		assoc_y.push_back(predicted[min_dist_idx].y);
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
	/** Step 1: Transform the observation coordinates into the world frame
	 * Step 2: Find only the relevant landmark set in the vicinity of each particle
	 * Step 3: Find the landmark closest to each of the observations
	 * Step 4: Calculate each particle's weight based on the product of multivariate Gaussian distributions of distances between the
	 * projected observations and mapped landmarks
	 */

	// Calculate the first term of the mvGd
	double gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);

	// Calculate the denominators of the mvGd
	double denom1 = 2 * pow(std_landmark[0], 2);
	double denom2 = 2 * pow(std_landmark[1], 2);

	// Iterate over all the particles
	for (int i = 0; i < num_particles; i++){
		// Initialize the weight product for each particle
		double weight_product = 1.0;
		vector<LandmarkObs> transformed_obs;
		/** Step 1
		 */
		// Transform the observation coordinates from the vehicle coordinate system to the map coordinate system
		for (int j = 0; j < observations.size(); j++){

			double x_transf = particles[i].x + (cos(particles[i].theta) * observations[j].x) - (sin(particles[i].theta) * observations[j].y);
			double y_transf = particles[i].y + (sin(particles[i].theta) * observations[j].x) + (cos(particles[i].theta) * observations[j].y);
			transformed_obs.push_back( LandmarkObs{ observations[j].id, x_transf, y_transf } );
		}

		/** Step 2
		 */
		// Find the landmarks within the vicinity of this particle
		vector<LandmarkObs> valid_landmarks;

		for (int k = 0; k < map_landmarks.landmark_list.size(); k++){

			// Calculate the distance between the particle and the landmarks
			double dist_particle2lm = sqrt(pow(double(map_landmarks.landmark_list[k].x_f) - particles[i].x, 2)
					+ pow(double(map_landmarks.landmark_list[k].y_f) - particles[i].y, 2));

			// Save the landmarks that are within the distance of a sensor range for this particle
			if (dist_particle2lm <= sensor_range){
				valid_landmarks.push_back( LandmarkObs{ map_landmarks.landmark_list[k].id_i, double(map_landmarks.landmark_list[k].x_f),
					double(map_landmarks.landmark_list[k].y_f) });
			}
		}

		/** Step 3
		 */
		vector<int> associations;
		vector<double> sense_x;
		vector<double> sense_y;
		vector<double> assoc_x;
		vector<double> assoc_y;

		// Associate observations with the landmarks
		dataAssociation(valid_landmarks, transformed_obs, associations, sense_x, sense_y, assoc_x, assoc_y);
		// Fill in the association variables to enable the associations' visualization
		SetAssociations(particles[i], associations, sense_x, sense_y);

		/** Step 4
		 */
		// Calculate multi-variate Gaussian distribution
		for (int m = 0; m < transformed_obs.size(); m++){
			double b = (pow(transformed_obs[m].x - assoc_x[m], 2) / denom1)
					+ (pow(transformed_obs[m].y - assoc_y[m], 2) / denom2);
			weight_product *= gauss_norm * exp(-b);
		}
		// Update particle weights with combined multi-variate Gaussian distribution
		particles[i].weight = weight_product;
		weights[i] = particles[i].weight;
	}
}


void ParticleFilter::resample() {

	vector<Particle> particles_resampled (num_particles);
	// Initialize the random number generator
	std::random_device rd;
	std::default_random_engine gen(rd());
	std::discrete_distribution<int> d(weights.begin(), weights.end());
	for (int i = 0; i < num_particles; i++){
		// Draw a random index
		int idx = d(gen);
		// Draw a particle with a corresponding index
		particles_resampled[i] = particles[idx];
	}

	// Update the particles with the resampled population
	particles = particles_resampled;
}



void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
