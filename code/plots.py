import matplotlib.pyplot as plt


def plot_membrane_potential(net, ng, ng_rec_name, title):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.plot(net[ng_rec_name, 0].variables["u"][:, :3], label='potential')
    ax1.axhline(y=ng.behavior[3].init_kwargs['threshold'], color='r', linestyle='--', label='Threshold')
    ax1.axhline(y=ng.behavior[3].init_kwargs['u_reset'], color='black', linestyle='--', label='u_reset')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('u(t)')
    ax1.set_title(f'Membrane Potential')
    ax1.legend()

    ax2.plot(net[ng_rec_name, 0].variables["I"][:, :3], label="current")
    ax2.set_xlabel('Time')
    ax2.set_ylabel("I(t)")
    ax2.set_title('Current')
    ax2.legend()
    fig.suptitle(title)
    plt.tight_layout()

    plt.show()
