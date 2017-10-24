
# coding: utf-8

# # Subplots

# In[1]:

get_ipython().magic('matplotlib notebook')

import matplotlib.pyplot as plt
import numpy as np

get_ipython().magic('pinfo plt.subplot')


# In[2]:

plt.figure()
# subplot with 1 row, 2 columns, and current axis is 1st subplot axes
plt.subplot(1, 2, 1)

linear_data = np.array([1,2,3,4,5,6,7,8])

plt.plot(linear_data, '-o')


# In[23]:

exponential_data = linear_data**2 

# subplot with 1 row, 2 columns, and current axis is 2nd subplot axes
plt.subplot(1, 2, 2)
plt.plot(exponential_data, '-o')


# In[4]:

# plot exponential data on 1st subplot axes
plt.subplot(1, 2, 1)
plt.plot(exponential_data, '-x')


# In[5]:

plt.gcf().canvas.draw()


# In[6]:

plt.figure()
ax1 = plt.subplot(1, 2, 1)
plt.plot(linear_data, '-o')
# pass sharey=ax1 to ensure the two subplots share the same y axis
ax2 = plt.subplot(1, 2, 2, sharey=ax1)
plt.plot(exponential_data, '-x')


# In[7]:

plt.figure()
# the right hand side is equivalent shorthand syntax
plt.subplot(1,2,1) == plt.subplot(121)


# In[8]:

# create a 3x3 grid of subplots
fig, ((ax1,ax2,ax3), (ax4,ax5,ax6), (ax7,ax8,ax9)) = plt.subplots(3, 3, sharex=True, sharey=True)
# plot the linear_data on the 5th subplot axes 
ax5.plot(linear_data, '-')


# In[24]:

# set inside tick labels to visible
for ax in plt.gcf().get_axes():
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_visible(True)
        


# In[25]:

# necessary on some systems to update the plot
plt.gcf().canvas.draw()


# # Histograms

# In[11]:

# create 2x2 grid of axis subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True)
axs = [ax1,ax2,ax3,ax4]

# draw n = 10, 100, 1000, and 10000 samples from the normal distribution and plot corresponding histograms
for n in range(0,len(axs)):
    sample_size = 10**(n+1)
    sample = np.random.normal(size=sample_size)
    axs[n].hist(sample)
    axs[n].set_title('n={}'.format(sample_size))


# In[12]:

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True)
axs = [ax1, ax2, ax3, ax4]

for n in (range(len(axs))):
    sample_size = 10**(n+1)
    sample = np.random.normal(size=sample_size)
    axs[n].hist(sample, bins=100)
    axs[n].set_title("n = {}".format(sample_size))
for plot in plt.gcf().get_axes():
    for label in plot.get_xticklabels() + plot.get_yticklabels():
        label.set_visible(True)
plt.subplots_adjust(bottom=0.1,hspace=0.5, wspace=0.35)


# In[13]:

get_ipython().magic('pinfo plt.subplots_adjust')


# In[14]:

# repeat with number of bins set to 100
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True)
axs = [ax1,ax2,ax3,ax4]

for n in range(0,len(axs)):
    sample_size = 10**(n+1)
    sample = np.random.normal(loc=0.0, scale=1.0, size=sample_size)
    axs[n].hist(sample, bins=100)
    axs[n].set_title('n={}'.format(sample_size))


# In[26]:

plt.figure()
Y = np.random.normal(loc=0.0, scale=1.0, size=10000)
X = np.random.random(size=10000)
plt.scatter(X,Y)


# In[29]:

get_ipython().magic('pinfo plt.subplot')


# In[30]:

# use gridspec to partition the figure into subplots
import matplotlib.gridspec as gridspec

plt.figure()
gspec = gridspec.GridSpec(3, 3)

top_histogram = plt.subplot(gspec[0, 1:])
side_histogram = plt.subplot(gspec[1:, 0])
lower_right = plt.subplot(gspec[1:, 1:])
plt.subplots_adjust(wspace=0.4, hspace=0.4)


# In[33]:

Y = np.random.normal(loc=0.0, scale=1.0, size=10000)
X = np.random.random(size=10000)
lower_right.scatter(X, Y)
top_histogram.hist(X, bins=100)
s = side_histogram.hist(Y, bins=100, orientation='horizontal')


# In[34]:

get_ipython().magic('pinfo top_histogram.hist')


# In[35]:

# clear the histograms and plot normed histograms
top_histogram.clear()
top_histogram.hist(X, bins=100, normed=True)
side_histogram.clear()
side_histogram.hist(Y, bins=100, orientation='horizontal', normed=True)
# flip the side histogram's x axis
side_histogram.invert_xaxis()


# In[36]:

# change axes limits
for ax in [top_histogram, lower_right]:
    ax.set_xlim(0, 1)
for ax in [side_histogram, lower_right]:
    ax.set_ylim(-5, 5)


# In[37]:

get_ipython().run_cell_magic('HTML', '', "<img src='http://educationxpress.mit.edu/sites/default/files/journal/WP1-Fig13.jpg' />")


# # Box and Whisker Plots

# In[38]:

get_ipython().magic('pinfo np.random.gamma')


# In[39]:

import pandas as pd
normal_sample = np.random.normal(loc=0.0, scale=1.0, size=10000)
random_sample = np.random.random(size=10000)
gamma_sample = np.random.gamma(3, size=10000)

df = pd.DataFrame({'normal': normal_sample, 
                   'random': random_sample, 
                   'gamma': gamma_sample})


# In[40]:

df.describe()


# In[41]:

get_ipython().magic('pinfo plt.boxplot')


# In[42]:

plt.figure()
# create a boxplot of the normal data, assign the output to a variable to supress output
_ = plt.boxplot(df['normal'], whis='range')


# In[43]:

# clear the current figure
plt.clf()
# plot boxplots for all three of df's columns
_ = plt.boxplot([ df['normal'], df['random'], df['gamma'] ], whis='range')


# In[44]:

_


# In[45]:

plt.figure()
_ = plt.hist(df['gamma'], bins=100)


# In[46]:

import mpl_toolkits.axes_grid1.inset_locator as mpl_il

plt.figure()
plt.boxplot([ df['normal'], df['random'], df['gamma'] ], whis='range')
# overlay axis on top of another 
ax2 = mpl_il.inset_axes(plt.gca(), width='60%', height='40%', loc=2, borderpad=3)
ax2.hist(df['gamma'], bins=100)
ax2.margins(x=0.5)


# In[47]:

get_ipython().magic('pinfo mpl_il.inset_axes')


# In[48]:

# switch the y axis ticks for ax2 to the right side
ax2.yaxis.tick_right()


# In[49]:

# if `whis` argument isn't passed, boxplot defaults to showing 1.5*interquartile (IQR) whiskers with outliers
plt.figure()
_ = plt.boxplot([ df['normal'], df['random'], df['gamma'] ] )


# # Heatmaps

# In[50]:

plt.figure()

Y = np.random.normal(loc=0.0, scale=1.0, size=10000)
X = np.random.random(size=10000)
_ = plt.hist2d(X, Y, bins=25)


# In[51]:

plt.figure()
_ = plt.hist2d(X, Y, bins=100)


# In[52]:

# add a colorbar legend
plt.colorbar()


# # Animations

# In[61]:

import matplotlib.animation as animation

n = 100
x = np.random.randn(n)


# In[65]:

# create the function that will do the plotting, where curr is the current frame
def update(curr):
    # check if animation is at the last frame, and if so, stop the animation a
    if curr == n: 
        a.event_source.stop()
    plt.cla()
    bins = np.arange(-4, 4, 0.5)
    plt.hist(x[:curr], bins=bins)
    plt.axis([-4,4,0,30])
    plt.gca().set_title('Sampling the Normal Distribution')
    plt.gca().set_ylabel('Frequency')
    plt.gca().set_xlabel('Value')
    plt.annotate('n = {}'.format(curr), [3,27])
    


# In[66]:

get_ipython().magic('pinfo animation.FuncAnimation')


# In[67]:

fig = plt.figure()
a = animation.FuncAnimation(fig, update, interval=100)


# # Interactivity

# In[68]:

plt.figure()
data = np.random.rand(10)
plt.plot(data)

def onclick(event):
    plt.cla()
    plt.plot(data)
    plt.gca().set_title('Event at pixels {},{} \nand data {},{}'.format(event.x, event.y, event.xdata, event.ydata))

# tell mpl_connect we want to pass a 'button_press_event' into onclick when the event is detected
plt.gcf().canvas.mpl_connect('button_press_event', onclick)


# In[69]:

from random import shuffle
origins = ['China', 'Brazil', 'India', 'USA', 'Canada', 'UK', 'Germany', 'Iraq', 'Chile', 'Mexico']

shuffle(origins)

df = pd.DataFrame({'height': np.random.rand(10),
                   'weight': np.random.rand(10),
                   'origin': origins})
df


# In[89]:

plt.figure()
# picker=5 means the mouse doesn't have to click directly on an event, but can be up to 5 pixels away
plt.scatter(df['height'], df['weight'], picker=5)
plt.gca().set_ylabel('Weight')
plt.gca().set_xlabel('Height')


# In[90]:

def onpick(event):
    #print(event.ind[0])
    origin = df.iloc[event.ind[0]]['origin']
    plt.gca().set_title('Selected item came from {}'.format(origin))
    
# tell mpl_connect we want to pass a 'pick_event' into onpick when the event is detected
plt.gcf().canvas.mpl_connect('pick_event', onpick)


# In[ ]:



