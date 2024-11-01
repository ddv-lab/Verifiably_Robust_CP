from sim import MPESim
from util.args import get_args

class MPERunner:
    
    def __init__(self, args):
        self.args = args
        self.sim = MPESim(self.args)

    def train(self):
        print("[INFO] Training environment and saving policies")
        result, agents = self.sim.train_agents()

    def test(self):
        print("[INFO] Evaluating environment and saved policies")
        self.sim.run_simulator()
#self.sim.output_observations(n_episodes=1)

    def generate(self):
        print("[INFO] Generating trajectories")
        self.sim.generate_trajectories()

    def run(self):
        self.train()
        self.generate()

    # TODO: Add checks for if the mode does not have required CLI args
    def parse_mode(self):
        try:
            launch = getattr(self, self.args.mode)
            launch()
        except AttributeError:
            print("[ERRO] Invalid mode provided! Please check and try again.")
            exit(1)
    

# Main execution entry point for package
def main():
    args = get_args()
    runner = MPERunner(args)
    runner.parse_mode()

if __name__ == "__main__":
    main()
