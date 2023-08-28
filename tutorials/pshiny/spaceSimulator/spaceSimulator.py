# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # imports


from shiny.types import NavSetArg
from typing import List
import openai
from shiny import App, Inputs, Outputs, Session, render, ui, reactive
from shiny.ui import TagList, div, h3, tags, head_content
import shinyswatch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date, datetime, timedelta
import requests
import asyncio
import astropy
from simulation import Body, Simulation, nbody_solve, spherical_to_cartesian
import astropy.units as u
import plotly.express as px
import API_keys_config

# Variable Setup

# DALL-E API key
openai.api_key = API_keys_config.openai_api_key

# NASA API Keys
DEMO_KEY = API_keys_config.NASA_api_key

# Base Folder Path
basePath = "/home/jovyan/shared/notebooks/spaceSimulator"

def getDate():
    return date.today()


print(getDate())


def url():
    return f"https://api.nasa.gov/neo/rest/v1/feed?start_date={getDate()}&api_key={DEMO_KEY}"


print(url())

dir = str(Path(basePath).absolute())
assets = dir + "/assets",
css_file = dir + "/assets/css/styles.css",
css_text = open(basePath + "/assets/css/styles.css", "r")
js_text = open(basePath + "/assets/js/script.js", "r")


async def fetchData():
    response = requests.get(url())
    if response.status_code != 200:
        raise Exception(f"Error fetching {url()}: {response.status}")
        
    data = response.json()
    return data


def count_hazards(data):
    hazards = 0
    near_earth_objects = data["near_earth_objects"].get(getDate().strftime('%Y-%m-%d'), [])
    for curr in near_earth_objects:
        if curr.get("is_potentially_hazardous_asteroid", True):
            hazards += 1
    return hazards


def result_hazards(data):
    near_earth_objects = data["near_earth_objects"].get(getDate().strftime('%Y-%m-%d'), [])
    return near_earth_objects


async def index():
    data = await fetchData()
    hazards = count_hazards(data)
    print(hazards)


# +
YesNo = {
    True: ui.p('YES 😱', class_="d-flex alert alert-danger rounded p-2 w-100 justify-content-center m-auto"),
    False: ui.p('nope',class_="d-flex border border-primary rounded p-2 w-100 justify-content-center m-auto")
}

def hazard(yes):
    return YesNo[yes]


# -

# Function for number formatting
def format_number(number):
    return "{:,.0f}".format(number)


asteroid_speeds = []


def passing(data):
    for i, item in enumerate(data):
        orbiting_body = item["orbiting_body"]
        date_close_approach = datetime.strptime(item["close_approach_date_full"], "%Y-%b-%d %H:%M")
        miss_distance_miles = float(item["miss_distance"]["miles"])
        relative_velocity_mph = float(item["relative_velocity"]["miles_per_hour"])

        formatted_date = date_close_approach.strftime('%d, %b %Y').lower()
        formatted_time = date_close_approach.strftime('%I:%M%p').lower()
        formatted_miss_distance = format_number(miss_distance_miles)
        formatted_relative_velocity = format_number(relative_velocity_mph)
        asteroid_speeds.append(formatted_relative_velocity)

        paragraph = f"Misses {orbiting_body} on {formatted_date} at {formatted_time} by {formatted_miss_distance} miles while travelling at {formatted_relative_velocity} mph"

    return paragraph


def orbital(name, is_potentially_hazardous_asteroid, close_approach_data, nasa_jpl_url):
    
    formatted_name = name.replace("(", "").replace(")", "")
    formatted_is_hazard = "YES 😱" if is_potentially_hazardous_asteroid else "nope"
    formatted_passing_data = passing(close_approach_data)
    
    return ui.div(
        ui.div(
            ui.div( 
                ui.h1(f"{formatted_name}", class_="py-3 display-3"),
                ui.div(
                    ui.h2(f"Potentially hazardous? ", class_="py-3 small"),
                    hazard(is_potentially_hazardous_asteroid),
                    class_="d-flex flex-row ",
                ),
                class_="d-flex flex-column w-50 m-4 p-4",
            ),
            ui.div(
                ui.p(formatted_passing_data, class_="d-flex p-3 "),
                class_="d-flex flex-column w-50 m-4 p-4 justify-content-center",
            ),
            class_="d-flex",
        ),
        ui.div(
            ui.a("Find out more", href=nasa_jpl_url, target="_blank", class_=""),
            class_="d-flex justify-content-center",
        ),
        class_="d-flex flex-column border border-primary rounded m-4 p-4 ",
    )


# Sort data by potential Danger
def format_data(data):
    sorted_results = sorted(data, key=lambda item: not item['is_potentially_hazardous_asteroid'])
    return sorted_results


def emoji(hazards):
    return '😱' if hazards > 0 else '👍'


# # Shiny App UI


def nav_controls(prefix: str) -> List[NavSetArg]:
    return [
        ui.nav("Is Earth in Danger? 🌎 ☄ 😱", 
            ui.div(
                ui.div(
                    ui.p('Select a date to see the potential impact of asteroids on Earth.'),
                    class_="d-flex flex-row justify-content-center"
                ),
                ui.div(
                    ui.input_action_button("today", "Today", class_="btn btn-outline-dark btn-lg p-2 m-2 rounded buttons"),
                    ui.input_action_button("tomorrow", "Tomorrow ",class_="btn btn-outline-dark btn-lg p-2 m-2 rounded buttons"),
                    ui.div(
                        ui.input_date("date"," ", format="yyyy-mm-dd"),
                        class_="btn btn-outline-dark p-2 m-2 rounded"                    
                    ),
                    class_="d-flex flex-row justify-content-center card rounded m-3"
                ),
            ),
            {"id": "main-content"},
            ui.output_ui("app", class_="asteroid d-flex flex-column"),
        ),
        ui.nav("Annihilation Simulator 👽",
            ui.div (
                ui.h2("Simulate Chaos", class_="text-center"),
                ui.p("Simulate earth, moon, and asteroid orbit, and see what Dall-e AI thinks the simulation looks like", class_="text-center"),
                ui.p("Adjust the cursor on the image to display the simulation plot or the AI generated image.", class_="text-center"),
                class_="d-flex flex-column mb-2 justify-content-center"
            ),
            ui.layout_sidebar(
                ui.panel_sidebar(
                    ui.p("Change the variables, run the simulation and see the changes"),
                    ui.input_slider("days", "Simulation duration (days)", 0, 200, value=60),
                    ui.input_slider(
                        "step_size",
                        "Simulation time step (hours)", 0, 24, value=4, step=0.5,
                    ),
                    ui.input_action_button(
                        "run", "Run simulation", class_="btn-outline-danger rounded w-100"
                    ),
                    ui.navset_tab_card(
                        # earth
                        ui.nav(
                            "Earth",
                            ui.input_checkbox("earth", "Enable", True),
                            ui.panel_conditional(
                                "input.earth",
                                ui.input_numeric(
                                    "earth_mass",
                                    "Mass (10^22 kg)",
                                    597.216,
                                ),
                                ui.input_slider(
                                    "earth_speed",
                                    "Speed (km/s)",
                                    0,
                                    1,
                                    value=0.0126,
                                    step=0.001,
                                ),
                                ui.input_slider("earth_theta", "Angle (𝜃)", 0, 360, value=270),
                                ui.input_slider("earth_phi", "𝜙", 0, 180, value=90),
                            ),
                        ),
                        # Moon
                        ui.nav(
                            "Moon",
                            ui.input_checkbox("moon", "Enable", True),
                            ui.panel_conditional(
                                "input.moon",
                                ui.input_numeric("moon_mass", "Mass (10^22 kg)", 7.347),
                                ui.input_slider(
                                    "moon_speed", "Speed (km/s)", 0, 2, value=1.022, step=0.001
                                ),
                                ui.input_slider("moon_theta", "Angle (𝜃)", 0, 360, value=90),
                                ui.input_slider("moon_phi", "𝜙", 0, 180, value=90),
                            ),
                        ),
                        # Asteroid
                        ui.nav(
                            "Asteroid",
                            ui.input_checkbox("asteroid", "Enable", False),
                            ui.panel_conditional(
                                "input.asteroid",
                                ui.input_numeric("asteroid_mass", "Mass (10^22 kg)", 0.000347),
                                ui.input_slider(
                                    "asteroid_speed", "Speed (km/s)", 0, 50, value=3.022, step=0.1
                                ),
                                ui.input_slider("asteroid_theta", "Angle (𝜃)", 0, 360, value=90),
                                ui.input_slider("asteroid_phi", "𝜙", 0, 180, value=90),
                            ),
                        ),
                    ),
                    class_="bg-light"
                ),
                ui.column(
                    8,
                    ui.div(
                        ui.div(
                        ),
                        ui.div(
                            ui.div(
                                ui.output_ui("dalle_image", width="800px", height="800px"),
                                class_="img background-img"
                            ),
                            ui.div(
                                ui.output_plot("orbits", width="800px", height="800px"),
                                class_="img foreground-img ",
                            ),
                            tags.input(type="range", min=1, max=100, value=50, class_="slider", name='slider', id="slider"),
                            class_="border border-primary rounded img-comp-container justify-content-center",
                        ),
                    ),
                    class_="p-3",
                ),
            ),
        ),
    ]


app_ui = ui.page_navbar(
    *nav_controls("page_navbar"),
    title="🌌 Space Funky 💫 🪐",
    bg="#0062cc",
    inverse=True,
    footer=ui.div(
        {"style": "width:80%;margin: 0 auto"},
        ui.tags.style(css_text.read()),
        ui.tags.script(js_text.read()),
        ui.tags.style(
            """
            h4 {
                margin-top: 3em;
            }
            """
        ),
    )
)


# # Server Function

def server(input, output, session):
    
    # Reactive Value
    # Update the value of the data when button is pressed
    date_input = reactive.Value(date.today())
    
    @reactive.Effect
    @reactive.event(input.today, ignore_none=True)
    def today():
        date_input.set(date.today())
        
    @reactive.Effect
    @reactive.event(input.tomorrow, ignore_none=True)
    def tomorrow():
        date_input.set(date.today()  + timedelta(days=1))
    
    @reactive.Effect
    @reactive.event(input.date, ignore_none=True)
    def _():
        date_input.set(input.date())
    
    document_title = reactive.Value(
        ui.TagList(
            ui.h1("Counting potential earth HAZARDS…"),
            ui.hr(),
            ui.p("Getting data from NASA right now to check whether something from space is going to hit us. One moment…"),
        )
    )
    
    @reactive.Effect
    def set_document():
        document_title = reactive.Value(
            ui.TagList(
                ui.h1("Counting potential earth HAZARDS…"),
                ui.hr(),
                ui.p("Getting data from NASA right now to check whether something from space is going to hit us. One moment…"),
            )
        )
    
    def update_document(results, hazards, orbital_results):
        document_title.set(
            ui.TagList(
                ui.div(
                    ui.h2(f"On {date_input.get().strftime('%A %d, %b')}; {len(results)} near misses and", 
                      class_="d-flex justify-content-center"),
                    ui.div(
                        ui.h3(f"{hazards} potential HAZARDS {emoji(hazards)}", 
                      class_="d-flex justify-content-center rounded alert alert-danger m-4 p-4 w-5"),
                       class_="d-flex justify-content-center mx-4 px-4" 
                    ),
                    class_="d-flex flex-column"
                ),
                ui.div(orbital_results, class_="h4")
            )
        )
    
    @reactive.Calc
    async def fetchData():
        response = requests.get(f"https://api.nasa.gov/neo/rest/v1/feed?start_date={date_input.get()}&api_key={DEMO_KEY}")
        if response.status_code != 200:
            raise Exception(f"Error fetching {url()}: {response.status}")
        data = response.json()
        return data

    @output
    @render.ui
    async def app():
        data = await fetchData()
        document_title.set(
            ui.TagList(
                ui.h1("Counting potential earth HAZARDS…"),
                ui.hr(),
                ui.p("Getting data from NASA right now to check whether something from space is going to hit us. One moment…"),
            )
        )
        # Get count on hazardous objects
        hazards = count_hazards(data)
        results = result_hazards(data)
        orbital_results = [orbital(value["name"], value["is_potentially_hazardous_asteroid"], 
                                   value["close_approach_data"], value["nasa_jpl_url"]) for value in format_data(result_hazards(data))]
        # asteroid_prompt.append(f"moon orbit around the sun")

        update_document(results, hazards, orbital_results)

        return document_title.get()

    # Simulation
    def earth_body():
        v = spherical_to_cartesian(
            input.earth_theta(), input.earth_phi(), input.earth_speed()
        )
        return Body(
            mass=input.earth_mass() * 10e21 * u.kg,
            x_vec=np.array([0, 0, 0]) * u.km,
            v_vec=np.array(v) * u.km / u.s,
            name="Earth",
        )
    
    def moon_body():
        v = spherical_to_cartesian(
            input.moon_theta(), input.moon_phi(), input.moon_speed()
        )
        return Body(
            mass=input.moon_mass() * 10e21 * u.kg,
            x_vec=np.array([3.84e5, 0, 0]) * u.km,
            v_vec=np.array(v) * u.km / u.s,
            name="Moon",
        )
    
    def asteroid_body():
        v = spherical_to_cartesian(
            input.asteroid_theta(), input.asteroid_phi(), input.asteroid_speed()
        )

        return Body(
            mass=input.asteroid_mass() * 10e21 * u.kg,
            x_vec=np.array([-3.84e5, 0, 0]) * u.km,
            v_vec=np.array(v) * u.km / u.s,
            name="Asteroid",
        )

    def simulation():
        bodies = []
        asteroid_prompt = "The "
        if input.earth():
            bodies.append(earth_body())
            asteroid_prompt += "earth "
        if input.moon():
            bodies.append(moon_body())
            asteroid_prompt += "moon orbit  "
        if input.asteroid():
            bodies.append(asteroid_body())
            asteroid_prompt += "and an asteroid hurling towards earth "

        simulation_ = Simulation(bodies)
        simulation_.set_diff_eq(nbody_solve)

        return [simulation_, asteroid_prompt]
    
    @output
    @render.plot
    @reactive.event(input.run, ignore_none=False)
    def orbits():
        return make_orbit_plot()

    def make_orbit_plot():
        sim = simulation()[0]
        n_steps = input.days() * 24 / input.step_size()
        with ui.Progress(min=1, max=n_steps) as p:
            sim.run(input.days() * u.day, input.step_size() * u.hr, progress=p)

        sim_hist = sim.history
        end_idx = len(sim_hist) - 1

        fig = plt.figure()

        ax = plt.axes(projection="3d")

        n_bodies = int(sim_hist.shape[1] / 6)
        for i in range(0, n_bodies):
            ax.scatter3D(
                sim_hist[end_idx, i * 6],
                sim_hist[end_idx, i * 6 + 1],
                sim_hist[end_idx, i * 6 + 2],
                s=50,
            )
            ax.plot3D(
                sim_hist[:, i * 6],
                sim_hist[:, i * 6 + 1],
                sim_hist[:, i * 6 + 2],
            )

        ax.view_init(30, 20)
        set_axes_equal(ax)

        return fig

    
    # This output updates only when input.btn is invalidated.
    @output
    @render.ui
    @reactive.event(input.run, ignore_none=False)
    def dalle_image():
        response = openai.Image.create(
            prompt=simulation()[1],
            model="image-alpha-001",
            size="1024x1024",
            response_format="url"
        )
        return ui.img({"src": response["data"][0]["url"], "width": "800px", "height": "800px",})


www_dir = Path(__file__).parent / "assets"
app = App(app_ui, server, static_assets=www_dir)


# https://stackoverflow.com/a/31364297/412655
def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

