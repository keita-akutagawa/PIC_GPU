#include "../particle_struct.hpp"
#include "../const.hpp"
#include <gtest/gtest.h>
#include <iostream>


TEST(ParticleStruct, setParticle)
{
    const int totalNumParticles = 10;
    std::vector<Particle> particles(totalNumParticles);

    EXPECT_EQ(particles.size(), totalNumParticles);

    for (int i = 0; i < totalNumParticles; i++) {
        EXPECT_EQ(particles[i].x, 0.0);
    }
}


