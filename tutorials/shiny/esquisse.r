# Load libraries ----
library(shiny)
library(esquisse)
library(datamods) 
library(tidyverse)

# Define UI ----
ui <- navbarPage(
  title = "Navigation",
  
  # Import datamods
  tabPanel(
    title = "Load Data",
    fluidRow(
      htmlOutput("welcome"),
      tags$head(tags$style(HTML("
                                #welcome {
                                  text-align: center;
                                }
                                ")
      )
      ),
      hr()
    ),
    fluidRow(
      column(
        width = 4,
        checkboxGroupInput(
          inputId = "from",
          label = "Input Source",
          choices = c("env", "file", "copypaste", "googlesheets", "url"),
          selected = c("env", "file")
        ),
        actionButton("launch_modal", "Launch Data Modal Window")
      ),
      column(
        width = 8,
        htmlOutput("info")
      ),
      column(
        width = 8,
        hr(),
        htmlOutput("loaded"),
        verbatimTextOutput("summary"),
        tags$head(tags$style("#summary{max-width: 500px;}")) # sets max box window for scrolling
      )
    )
  ),
  
  # Output: View Data ----
  tabPanel(
    title = "View Data",
    htmlOutput("dtheader"),
    hr(),
    DT::dataTableOutput("contents")
    # tableOutput("contents")
  ),
  
  # Output: Esquisse ----
  tabPanel(
    title = "Plot (Esquisse)",
    htmlOutput("esqheader"),
    checkboxGroupInput(
      inputId = "aes", 
      label = "Aesthetic Options:", 
      choices = c(
        "fill", "color", "size", "shape", 
        "weight", "group", "facet", "facet_row", "facet_col"
      ),
      selected = c("fill", "color", "size", "shape", "facet"),
      inline = TRUE
    ),
    esquisse_ui(
      id = "esquisse", 
      header = FALSE, # set to TRUE to see blue esquisse icon header
      container = esquisseContainer(
        fixed = c(150, 0, 0, 0)
      )
    )
  )
)

server <- function(input, output, session) {
  
  output$welcome <- renderUI({
    str <- tags$h3("Welcome")
    str0 <- print("This app adapts code from add-ins and modules created by the dreamRs team at ")
    tag <- tags$a(href="https://github.com/dreamRs/", "github.com/dreamRs")
    spacer <- print(".")
    HTML(paste(str, str0, tag, spacer, sep = ""))
  })
  
  output$info <- renderUI({
    str1 <- print("<b>Instructions</b>")
    str2 <- print("- Select input source options to upload dataset.")
    str3 <- print("- Example datasets can be loaded through 'env' (global environment).")
    str4 <- print("- Once desired dataset is selected, variable classes can be changed as needed.")
    str5 <- print("&ensp; e.g., characters can be changed to factors if data is categorical.")
    str_launch <- print("<br>Click on <b>'Launch Data Modal Window'</b> to launch module window.")
    HTML(paste(str1, str2, str3, str4, str5, str_launch, sep = '<br/>'))
  })
  
  # Launch import data modal 
  observeEvent(input$launch_modal, {
    req(input$from)
    import_modal(
      id = "myid",
      from = input$from,
      title = "Import data to be used in application"
    )
  })
  data_imported_r <- import_server("myid", return_class = "tbl_df")
  # data_imported_r <- datamods::import_server("import-data")
  
  data_rv <- reactiveValues(data = data.frame())
  observeEvent(data_imported_r$data(), {
    data_rv$data <- data_imported_r$data()
    data_rv$name <- data_imported_r$name()
  })
  
  # Output name as part of summary ----
  output$loaded <- renderUI({
    str_ld <- print("<b>Loaded Dataset: </b>")
    str_fn <- print(data_rv$name) # dataset/file name
    HTML(paste(str_ld, str_fn, sep = ""))
  })
  
  
  # Generate a summary of the dataset ----
  output$summary <- renderPrint({
    if(nrow(data_rv$data)==0){
      # do nothing
    }
    else{
      summary(data_rv$data)
    }
    # print(names(data_rv$data)) # variable names
  })
  
  # dataset contents header ----
  output$dtheader <- renderUI({
    HTML("<b>Please note:</b> activating a column filter cannot be reset to neutral at this time.")
  })
  
  # Show contents of dataset ----
  output$contents <- DT::renderDataTable({
    data_rv$data %>% 
      DT::datatable(options = list(pageLength=10),
                    rownames=FALSE)
  })
  
  # Header for Esquisse ----
  output$esqheader <- renderUI({
    HTML("<b>Instructions:</b> Drag and drop variables into aesthetic boxes below to generate plots.")
  })
  
  # Launch Esquisse ----  
  esquisse_server(id = "esquisse", data_rv = data_rv, default_aes = reactive(input$aes))
  
}

# Launch RShiny App
if (interactive())
  shinyApp(ui, server)