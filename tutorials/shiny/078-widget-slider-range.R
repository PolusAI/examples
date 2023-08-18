library(shiny)

server <- function(input, output) {

  # You can access the value of the widget (as a vector of length 2)
  # with input$slider2, e.g.
  output$value <- renderPrint({ input$slider2 })

}


ui <- fluidPage(
    
  # Copy the line below to make a slider bar with a range
  sliderInput("slider2", label = h3("Slider Range"), min = 0, max = 100, 
    value = c(25, 75)),
  
  hr(),
  fluidRow(column(3, verbatimTextOutput("value")))
    
)


# Create Shiny app ----
shinyApp(ui = ui, server = server)