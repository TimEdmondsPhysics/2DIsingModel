import numpy as np
import random as rnd # random initial lattice
import matplotlib.pyplot as plt # plot
from matplotlib import colors # custom colors
import matplotlib.animation as animation # for time-changing plotting
from matplotlib.widgets import Slider, Button # for interactive plotting
import scipy.special as scpspc # for special exotic elliptical functions
from matplotlib.gridspec import GridSpec # for plotting
from matplotlib.lines import Line2D

# Next things to work on
# * finish off commenting
# * confirming that the analytical energy and magnetisation calculations are correct
# * when B = 0 change magneisation below critical temp to have analytical curve at
# +- rather than just the one (randomly will choose one orientation)
# * add option to allow turning on a graph or something for heat capacity
# * add option to allow commands when script is run
# (stuff like whether it saves a video file, length of simulation)
# * copy working hear and adapt so that the plotting can be done in VisPy

def my_wrap(s, N):
    # This function deals with periodic boundary conditions on the lattice
    # needs a slight modification of the mod function to work correctly
    if s < 0:
        m = ((s+1) % N) + N-1 
    else:
        m = s % N
    return m

class spinLattice:

    def __init__(self, init_type, Lx, Ly, J_1, J_2, k_B, T, mu, B, num_skip, is_saving):
        
        self.saving = is_saving  # flag to toggle saving on/off
        self.file_name_count = 0
        
        self.fig = plt.figure(figsize=(15,10))
        
        self.Lx = Lx # number of sites in x axis
        self.Ly = Ly # number of sites in y axis (total grid Lx*Ly)
        self.J_1 = J_1 # Interaction with nearest neighbours (directly up and
        # to the side)
        self.J_2 = J_2 # Interaction with next nearest neighbours (diagonals)
        self.k_B = k_B # Boltzmann's constant
        self.T = T # Temperature
        self.mu = mu # Magnetic permeability
        self.B = B # Magnetic flux density
        
        self.init_type = init_type # determins what the lattice initialises as
        self.construct_neighbours() # generate neighbours array (3D), for use
        # with J_1
        self.construct_next_neighbours() # generates
        # next-neighbours array (3D), for use with J_2
        self.construct_lattice() # create lattice array
        self.calculate_energy() # calculates the energy of the lattice
        self.calculate_magnetisation() # calculates the magnetisation of the
        # lattice
        self.calculate_analytic() # calculates the analytical solutions for
        # energy, magnetisation, etc. if available
        self.T_c = 2/np.log(1 + np.sqrt(2))
        
        self.num_counted = 0
        self.E_count = 0.0
        self.M_count = 0.0
        self.Epow2_count = 0.0
        self.Mpow2_count = 0.0
        self.Epow4_count = 0.0
        self.Mpow4_count = 0.0
        self.num_iterated = 0
        self.num_skip = num_skip
        self.step_number = 0
        self.monte_carlo_steps_per_frame = 200
    
    def construct_lattice(self):
        # build the lattice
        self.lattice = np.empty([self.Ly, self.Lx])
        if self.init_type == "random":
            for ii in range(0, self.Lx):
                for jj in range(0, self.Ly):
                    self.lattice[ii,jj] = rnd.choice([-1,1])
                    # either up or down spin
        else:
            pass
    
    def construct_neighbours(self):
        # create a 3D array of the neighbours of each position on the lattice
        # for computational efficiency
        self.neighbours = np.empty([self.Lx*self.Ly , 4, 2], dtype=int)
        # the first dimension is the position on the lattice we're calculating
        # neighbours for
        # the second dimension is the x coordinator of the 
        # neighbour (goes up, right, down left)
        for ii in range(0, self.Lx):
            for jj in range(0, self.Ly):
                lat_pos = ii + jj*self.Lx; # where upon the array the lattice
                # position is
                
                # Upwards neighbour
                self.neighbours[lat_pos, 0, 0] = int(my_wrap(ii+1,self.Lx))
                self.neighbours[lat_pos, 0, 1] = jj
                
                # Rightwards neighbour
                self.neighbours[lat_pos, 1, 0] = ii
                self.neighbours[lat_pos, 1, 1] = int(my_wrap(jj+1,self.Ly))
                
                # Downwards neighbour
                self.neighbours[lat_pos, 2, 0] = int(my_wrap(ii-1,self.Lx))
                self.neighbours[lat_pos, 2, 1] = jj
                
                # Leftwards neighbour
                self.neighbours[lat_pos, 3, 0] = ii
                self.neighbours[lat_pos, 3, 1] = int(my_wrap(jj-1,self.Ly))
                
    def construct_next_neighbours(self):
        # create a 3D array of the next-nearest neighbours of each position on the lattice
        # for computational efficiency
        self.next_neighbours = np.empty([self.Lx*self.Ly, 4, 2], dtype=int)
        for ii in range(0, self.Lx):
            for jj in range(0, self.Ly):
                lat_pos = ii + jj*self.Lx;
                # Right-up neighbour (1 up 1 right)
                self.next_neighbours[lat_pos, 0, 0] = int(my_wrap(ii+1,self.Lx))
                self.next_neighbours[lat_pos, 0, 1] = int(my_wrap(jj+1,self.Ly))
                
                # Right-down neighbour (1 down 1 right)
                self.next_neighbours[lat_pos, 1, 0] = int(my_wrap(ii-1,self.Lx))
                self.next_neighbours[lat_pos, 1, 1] = int(my_wrap(jj+1,self.Ly))
                
                # Left-up neighbour (1 up 1 left)
                self.next_neighbours[lat_pos, 2, 0] = int(my_wrap(ii+1,self.Lx))
                self.next_neighbours[lat_pos, 2, 1] = int(my_wrap(jj-1,self.Ly))

                # Left-down neighbour (1 down 1 left)
                self.next_neighbours[lat_pos, 3, 0] = int(my_wrap(ii-1,self.Lx))
                self.next_neighbours[lat_pos, 3, 1] = int(my_wrap(jj-1,self.Ly))
        
    def all_down(self):
        # sets the lattice to have all spins down
        self.lattice = np.ones(self.Lx, self.Ly)
    
    def all_up(self):
        # sets the lattice to have all spins up
        self.lattice = -1*np.ones(self.Lx, self.Ly)
    
    def calculate_energy(self):
        # calculates the energy of the lattice
        magEnergy = 0 # energy due to the magnetic field
        interEnergy_1 = 0 # energy due to nearest neighbours
        interEnergy_2 = 0 # energy due to next nearest neighbours
        for row in self.lattice:
            for item in row:
                magEnergy += item
        magEnergy *= -1*self.mu*self.B # magEnergy = (mu*B*(sum(spins)))
        for ii in range(0, self.Lx):
            for jj in range(0, self.Ly):
                for kk in range(0, 4):
                    lat_pos = ii + jj*self.Lx
                    interEnergy_1 += (self.lattice[ii,jj])*self.lattice[self.neighbours[lat_pos, kk, 0],self.neighbours[lat_pos, kk, 1]]
                    #interEnergy_1 has decreased energy if the spins are aligned (see negative sign below)
                    interEnergy_2 += (self.lattice[ii,jj])*self.lattice[self.next_neighbours[lat_pos, kk, 0],self.next_neighbours[lat_pos, kk, 1]]
                    #interEnergy_2 has decreased energy if the spins are aligned (see negative sign below)
        interEnergy_1 *= -0.5*self.J_1 # negative of the above sum times interaction strength
        interEnergy_2 *= -0.5*self.J_2 # negative of the above sum times interaction strength
        self.energy = magEnergy + interEnergy_1 + interEnergy_2 # sum of all energies
        
    def calculate_magnetisation(self):
        mag = 0.0
        for ii in range(0, self.Lx):
            for jj in range(0, self.Ly):
                mag += self.lattice[ii,jj] # magnetisation is sum of spins
        self.magnetisation = mag
        
    def delta_energy(self, ii, jj):
        lat_pos = ii + jj*self.Lx
        # calculates the change in energy if the spin in flipped
        del_E = 2*self.lattice[ii,jj]*self.mu*self.B # magnetisation part
        for kk in range(0, 4):
            # neighbours interaction part
            del_E += 2*self.lattice[ii,jj]*self.J_1*self.lattice[self.neighbours[lat_pos, kk, 0],self.neighbours[lat_pos, kk, 1]]
            # next-neighbours interactions part
            del_E += 2*self.lattice[ii,jj]*self.J_2*self.lattice[self.next_neighbours[lat_pos, kk, 0],self.next_neighbours[lat_pos, kk, 1]]
        return del_E
    
    def delta_magnetisation(self, ii, jj):
        # calculates the change in magnetisation if this spin was flipped
        del_M = -2*self.lattice[ii, jj]
        return del_M

    def reset_counter(self):
        self.num_counted = 0
        self.E_count = 0.0
        self.M_count = 0.0
        self.Epow2_count = 0.0
        self.Mpow2_count = 0.0
        self.Epow4_count = 0.0
        self.Mpow4_count = 0.0
        self.num_iterated = 0

    def monte_carlo_step(self):
        x_flip = rnd.randint(0,self.Lx-1) # random x position
        y_flip = rnd.randint(0,self.Ly-1) # random y position
        deltaE = self.delta_energy(x_flip, y_flip) # what would be the change
        # in energy from flipping this spin?
        deltaM = self.delta_magnetisation(x_flip, y_flip) # what would be
        # the change in magnetisation from flipping this spin?
        w = np.exp(-1*deltaE/(self.k_B*self.T)) # calculates w based on deltaE
        if w < 1.0:
            R = rnd.uniform(0,1)
            if R <= w:
                # trial point accepted and add to average
                self.lattice[x_flip,y_flip] *= -1
                self.energy += deltaE
                self.magnetisation += deltaM
                if self.num_iterated >= self.num_skip: # If passed num_skip values start averaging
                    self.num_counted += 1;
                    self.E_count += self.energy
                    self.M_count += self.magnetisation
                    #self.Epow2_count += pow(self.energy,2.0)
                    #self.Mpow2_count += pow(self.magnetisation,2.0)
                    #self.Epow4_count += pow(self.energy,4.0)
                    #self.Mpow4_count += pow(self.magnetisation,4.0)
                else:
                    pass
            else:
                # trial point not accepted and add to average
                if self.num_iterated >= self.num_skip: # If passed num_skip values start averaging
                    self.num_counted += 1;
                    self.E_count += self.energy;
                    self.M_count += self.magnetisation;
                    #self.Epow2_count += pow(self.energy,2.0)
                    #self.Mpow2_count += pow(self.magnetisation,2.0)
                    #self.Epow4_count += pow(self.energy,4.0)
                    #self.Mpow4_count += pow(self.magnetisation,4.0)
                else:
                    pass
        else:
            # trial point accepted and not added to average
            self.lattice[x_flip,y_flip] *= -1
            self.energy += deltaE
            self.magnetisation += deltaM
        self.num_iterated += 1
    
    def update_graph(self, frame):
        # saves the current lattice variables as they will change in this
        # function
        self.previous_energy = self.energy
        self.previous_magnetisation = self.magnetisation
        for k in range(0,self.monte_carlo_steps_per_frame):
            self.monte_carlo_step() # try and update the lattice
        self.calculate_analytic() # calculate new analytical values
        self.step_number += 1
        #self.T = (7*np.sin(frame/50) + 7)*np.exp(-1.0*frame/1000) + 0.1
        #print(self.T)
        self.grid_plot.set_data(self.lattice) # plot data on lattice
        # plot energy
        energy_line = self.ax_energy.plot([(self.step_number) - 1, self.step_number], [(1/(self.Lx * self.Ly))*self.previous_energy, (1/(self.Lx * self.Ly))*self.energy], 'b', linewidth = 1)[0]
        # plot analytical energy
        energy_analytic_line = self.ax_energy.plot([(self.step_number) - 1, self.step_number], [(1/(self.Lx * self.Ly))*self.previous_analytical_E, (1/(self.Lx * self.Ly))*self.analytic_E], 'r', linewidth = 1)[0]
        # plot magnetisation
        mag_line = self.ax_mag.plot([(self.step_number) - 1, self.step_number], [(1/(self.Lx * self.Ly))*self.previous_magnetisation, (1/(self.Lx * self.Ly))*self.magnetisation], 'b', linewidth = 1)[0]
        # plot analytical magnetisation
        if self.B == 0:
            mag_analytic_line = self.ax_mag.plot([(self.step_number) - 1, self.step_number], [(1/(self.Lx * self.Ly))*self.previous_analytical_M, (1/(self.Lx * self.Ly))*self.analytic_M], 'r', linewidth = 1)[0]
            mirror_mag_analytic_line = self.ax_mag.plot([(self.step_number) - 1, self.step_number], [(-1/(self.Lx * self.Ly))*self.previous_analytical_M, (-1/(self.Lx * self.Ly))*self.analytic_M], 'r', linewidth = 1)[0]
            self.mag_lines.append((mag_line, mag_analytic_line, mirror_mag_analytic_line))
        else:
            mag_analytic_line = self.ax_mag.plot([(self.step_number) - 1, self.step_number], [(1/(self.Lx * self.Ly))*self.previous_analytical_M, (1/(self.Lx * self.Ly))*self.analytic_M], 'r', linewidth = 1)[0]
            self.mag_lines.append((mag_line, mag_analytic_line))
        
        self.energy_lines.append((energy_line, energy_analytic_line))
        
        if self.step_number > 50: # when 50 values are calculated start moving
            # the graph along the time-axis
    
            oldest_e, oldest_e_analytic = self.energy_lines.pop(0)
            oldest_e.remove()
            oldest_e_analytic.remove()
            
            m_list = self.mag_lines.pop(0)
            if len(m_list) == 2:
                oldest_m, oldest_m_analytic = m_list
                oldest_m.remove()
                oldest_m_analytic.remove()
            elif len(m_list) == 3:
                oldest_m, oldest_m_analytic, oldest_mirror_mag_analytic_line = m_list
                oldest_m.remove()
                oldest_m_analytic.remove()
                oldest_mirror_mag_analytic_line.remove()
            
            self.ax_energy.set_xlim(self.step_number - 52, self.step_number + 2)
            self.ax_mag.set_xlim(self.step_number - 52, self.step_number + 2)
        else:
            pass
        if self.saving:
            self.writer.grab_frame()
        self.previous_analytical_E = self.analytic_E
        self.previous_analytical_M = self.analytic_M
        
    def update_temp(self, val):
        self.T = self.slider_temp.val
    
    def update_J1(self, val):
        self.J_1 = self.slider_J1.val
        self.calculate_energy()
        
    def update_J2(self, val):
        self.J_2 = self.slider_J2.val
        self.calculate_energy()
        if self.J_2 == 0:
            if self.B == 0:
                self.critical_temp_line.set_visible(True)
                self.critical_temp_label.set_visible(True)
        else:
            self.critical_temp_line.set_visible(False)
            self.critical_temp_label.set_visible(False)
        
    def update_B(self, val):
        self.B = self.slider_B.val
        self.calculate_energy()
        if self.B == 0:
            if self.J_2 == 0:
                self.critical_temp_line.set_visible(True)
                self.critical_temp_label.set_visible(True)
        else:
            self.critical_temp_line.set_visible(False)
            self.critical_temp_label.set_visible(False)
        
    def reset(self, event):
        self.slider_temp.reset()
        self.slider_J1.reset()
        self.slider_J2.reset()
        self.slider_B.reset()
        self.calculate_energy()
        
        
    def start_recording(self, event):
        if not(self.saving):
            self.saving = True
            self.file_name_count += 1
            self.button_save.color = 'lightgreen'
            self.button_save.hovercolor = 'limegreen'
            self.button_start_recording.color = 'limegreen'
            self.turn_recording_light_on()
            self.writer = animation.FFMpegWriter(fps=60, metadata=dict(artist='Tim'), bitrate=3000)
            self.writer.setup(self.fig, r"D:\interactive physics and maths simulations\spin lattice\recording" + str(self.file_name_count) + ".mp4", dpi=100)
            print("Recording started")
        
    def save_recording(self, event):
        self.turn_recording_light_off()
        if self.saving:
            self.button_save.color = 'grey'
            self.button_save.hovercolor = 'grey'
            self.button_start_recording.color = 'lightgreen'
            self.saving = False
            self.writer.finish()
            print("Video saved")
    
    def init_graph(self):

        self.gs = GridSpec(3,4,figure=self.fig)
        self.ax_grid = self.fig.add_subplot(self.gs[0:,0:-1])
        self.ax_energy = self.fig.add_subplot(self.gs[0,-1])
        self.ax_mag = self.fig.add_subplot(self.gs[1,-1])
        self.ax_grid.set_axis_off() # no axes visible, better looking plot
        self.ax_energy.set_ylabel("Energy")
        self.ax_mag.set_ylabel("Magnetisation")
        legend_colours = ['blue', 'red']
        legend_lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='solid') for c in legend_colours]
        self.ax_energy.legend(legend_lines, ['Energy', 'Onsager solution energy'], loc=(0.025,0.85), fontsize=8)
        self.ax_mag.legend(legend_lines, ['Magnetisation', 'Onsager solution magnetisation'], loc=(0.025,0.8), fontsize=8)
        self.fig.tight_layout()
        
        if self.saving:
            self.file_name_count += 1
            self.writer = animation.FFMpegWriter(fps=60, metadata=dict(artist='Tim'), bitrate=3000)
            self.writer.setup(self.fig, r"D:\interactive physics and maths simulations\spin lattice\recording" + str(self.file_name_count) + ".mp4", dpi=100)
            print("Recording started")
        
        # create discrete colormap
        #global myLattice
        
        # colours of the up and down spins
        cmap = colors.ListedColormap(['darkturquoise', 'darkorchid'])
        bounds = [-1,0,1]
        latt_norm = colors.BoundaryNorm(bounds, cmap.N)
        self.grid_plot = self.ax_grid.imshow(self.lattice, cmap=cmap, norm=latt_norm, animated=True)
        self.energy_lines = []
        self.mag_lines = []
        
        self.ax_temp = plt.axes([0.75, 0.30, 0.21, 0.025], facecolor='lightgrey')
        self.slider_temp = Slider(self.ax_temp, '$T$', 0.1, 10, valinit=T_init, initcolor='navy')
        self.slider_temp.vline.set_linewidth(3)
        self.slider_temp.label.set_size(20)
        self.critical_temp_line = self.slider_temp.ax.axvline(self.T_c, color='red', linestyle='solid', linewidth=3, label='T_c')
        self.critical_temp_label = self.ax_temp.text(2, 0.43, "$T_C$", fontsize=14)
        self.slider_temp.on_changed(self.update_temp)

        self.ax_J1 = plt.axes([0.75, 0.25, 0.21, 0.025], facecolor='lightgrey')
        self.slider_J1 = Slider(self.ax_J1, '$J_1$', -5, 5, valinit=J_1_init, initcolor='navy')
        self.slider_J1.vline.set_linewidth(3)
        self.slider_J1.label.set_size(20)
        self.slider_J1.on_changed(self.update_J1)

        self.ax_J2 = plt.axes([0.75, 0.2, 0.21, 0.025], facecolor='lightgrey')
        self.slider_J2 = Slider(self.ax_J2, '$J_2$', -5, 5, valinit=J_2_init, initcolor='navy')
        self.slider_J2.vline.set_linewidth(3)
        self.slider_J2.label.set_size(20)
        self.slider_J2.on_changed(self.update_J2)

        self.ax_B = plt.axes([0.75, 0.15, 0.21, 0.025], facecolor='lightgrey')
        self.slider_B = Slider(self.ax_B, '$B$', -2, 2, valinit=B_init, initcolor='navy')
        self.slider_B.vline.set_linewidth(3)
        self.slider_B.label.set_size(20)
        self.slider_B.on_changed(self.update_B)

        self.reset_button = plt.axes([0.9, 0.06, 0.07, 0.07])
        self.button_reset = Button(self.reset_button, 'Reset\nSliders', color='skyblue', hovercolor='deepskyblue')
        self.button_reset.on_clicked(self.reset)
        
        if self.saving:
            self.start_colour = 'limegreen'
            self.save_colour = 'lightgreen'
            self.save_hover_colour = 'limegreen'
        else:
            self.start_colour = 'lightgreen'
            self.save_colour = 'grey'
            self.save_hover_colour = 'grey'

        self.save_button = plt.axes([0.82, 0.06, 0.07, 0.07])
        self.button_save = Button(self.save_button, 'Save\nRecording', color=self.save_colour, hovercolor=self.save_hover_colour)
        self.button_save.on_clicked(self.save_recording)

        self.start_recording_button = plt.axes([0.74, 0.06, 0.07, 0.07])
        self.button_start_recording = Button(self.start_recording_button, 'Start\nRecording', color=self.start_colour, hovercolor='limegreen')
        self.button_start_recording.on_clicked(self.start_recording)
        
        self.recording_light_axis = plt.axes([0.69, 0.065, 0.06, 0.06])
        outer_circle_plot = plt.Circle((0.5, 0.5), 0.2, color='maroon', fill=True)
        inner_circle_on_plot = plt.Circle((0.5, 0.5), 0.15, color='red', fill=True)
        inner_circle_off_plot = plt.Circle((0.5, 0.5), 0.15, color='xkcd:dried blood', fill=True)
        self.recording_light_outer = self.recording_light_axis.add_patch(outer_circle_plot)
        self.recording_light_inner_on = self.recording_light_axis.add_patch(inner_circle_on_plot)
        self.recording_light_inner_off = self.recording_light_axis.add_patch(inner_circle_off_plot)
        self.recording_light_axis.set_aspect('equal')
        self.recording_light_axis.set_axis_off()
        
        if self.saving:
            self.turn_recording_light_on()
        else:
            self.turn_recording_light_off()
            
        self.previous_analytical_E = self.analytic_E
        self.previous_analytical_M = self.analytic_M
            
    def turn_recording_light_on(self):
        self.recording_light_inner_on.set_visible(True)
        self.recording_light_inner_off.set_visible(False)
        
    def turn_recording_light_off(self):
        self.recording_light_inner_on.set_visible(False)
        self.recording_light_inner_off.set_visible(True)
        
    def calculate_analytic(self):
        if self.B == 0 and self.J_2 == 0:
            gamma = 2*self.J_1/(self.k_B*self.T)
            z = np.exp(-gamma)
            csch = 1/np.sinh(gamma)
            kappa = 4*(csch**2)/(csch**2+1)**2
            kappaPrime = 2*(np.tanh(gamma))**2 - 1
            K1_kappa = scpspc.ellipk(kappa)
            E1_kappa = scpspc.ellipe(kappa)
            self.analytic_E = self.Lx*self.Ly*(-1*self.J_1*(1/np.tanh(gamma))*(1 + (2/np.pi)*kappaPrime*K1_kappa))
            self.analytic_C = self.Lx*self.Ly*(2/np.pi)*(0.5*gamma/np.tanh(gamma))**2 * (2*K1_kappa - 2*E1_kappa - (1 - kappaPrime)*(np.pi/2 + kappaPrime*K1_kappa))
            if z < np.sqrt(2) - 1:
                self.analytic_M = self.Lx * self.Ly * (1 - (np.sinh(gamma)**2)**-2)**(0.125)
            else:
                self.analytic_M = 0
        elif self.J_2 == 0 and self.J_1 == 0:
            self.analytic_E = self.Lx * self.Ly * -1*self.mu*self.B*np.tanh(self.mu*self.B/(self.k_B*self.T))
            self.analytic_M = self.mu*np.tanh(self.mu*self.B/(self.k_B*self.T))
            self.analytic_C = 0.0
        else:
            self.analytic_E = 0.0
            self.analytic_C = 0.0
            self.analytic_M = 0.0
    
#     def calculate_analytic(self): # old plot
#         if self.B == 0 and self.J_2 == 0:
#             gamma = 2*self.J_1/(self.k_B*self.T)
#             z = np.exp(-gamma)
#             kappa = 2*np.sinh(gamma)/((np.cosh(gamma)**2))
#             kappaPrime = 2*(np.tanh(gamma))**2 - 1
#             K1_kappa = scpspc.ellipk(kappa)
#             E1_kappa = scpspc.ellipe(kappa)
#             self.analytic_E = self.Lx*self.Ly*-1*self.J_1*(np.cosh(gamma)/np.sinh(gamma))*(1 + (2/np.pi)*kappaPrime*K1_kappa)
#             self.analytic_C = self.Lx*self.Ly*(2/np.pi)*(0.5*gamma/np.tanh(gamma))**2 * (2*K1_kappa - 2*E1_kappa - (1 - kappaPrime)*(np.pi/2 + kappaPrime*K1_kappa))
#             if z < np.sqrt(2) - 1:
#                 self.analytic_M = self.Lx * self.Ly * ((1 + z**2)**(0.25)) * ((1 - 6*z**2) + z**4)**0.125 * (1 - z**2)**(-0.5)
#             else:
#                 self.analytic_M = 0
#         elif self.J_2 == 0 and self.J_1 == 0:
#             self.analytic_E = self.Lx * self.Ly * -1*self.mu*self.B*np.tanh(self.mu*self.B/(self.k_B*self.T))
#             self.analytic_M = self.mu*np.tanh(self.mu*self.B/(self.k_B*self.T))
#             self.analytic_C = 0.0
#         else:
#             self.analytic_E = 0.0
#             self.analytic_C = 0.0
#             self.analytic_M = 0.0
    
rnd.seed(12345) # seeds the random seed
x_length_init = 30 # x-grid length
y_length_init = 30 # y-grid length
J_1_init = 1
J_2_init = 0
k_B_init = 1
T_init = 1
mu_init = 1
B_init = 0
num_skip_init = 0
init_type = "random"
is_saving = False;

#fig, axes = plt.subplots(nrows=3, ncols=4)
#gs = axes[0,0].get_gridspec()
#for ax in axes[0:-1,0:2]:
#    ax.remove()
#fig.tight_layout()
#plt.subplots_adjust(left=0, bottom=0.45)
#ax1.position = [0.4,0.4,0.7,0.7]

myLattice = spinLattice(init_type, x_length_init, y_length_init, J_1_init, J_2_init, k_B_init, T_init, mu_init, B_init, num_skip_init, is_saving)

#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=30, metadata=dict(artist='Tim'), bitrate=1000)

anim = animation.FuncAnimation(myLattice.fig, myLattice.update_graph , init_func = myLattice.init_graph, interval  = 1, frames  = 10000, repeat=False)
plt.show()

#anim.save('im.mp4', writer=writer)