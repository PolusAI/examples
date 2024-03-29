library(shiny)

ui <- basicPage(
    actionButton("show", "Show modal dialog")
  )
  
server <- function(input, output) {
    observeEvent(input$show, {
      showModal(modalDialog(
        title = "Important message",
        "This is an important message!",
        easyClose = TRUE
      ))
    })
  }



# Create Shiny app ----
shinyApp(ui = ui, server = server)