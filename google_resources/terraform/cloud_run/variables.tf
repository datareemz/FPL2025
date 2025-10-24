variable "project_id" {
  type        = string
  description = "Your Google Cloud project ID."
}

variable "region" {
  type        = string
  description = "The GCP region (e.g., 'us-central1')."
}

variable "deletion_protection" {
  type    = bool
  default = false
}

variable "job_name" {
  type        = string
  description = "The name of the Cloud Run job."
  default     = "scheduled-fpl-2025-job"
}

variable "container_image" {
  type        = string
  description = "The container image to deploy."
}

variable "api_services_dependency" {
  description = "Dependency on API services being enabled"
  type        = any
  default     = null
}

variable "service_account_email" {
  type        = string
  description = "The service account email to use for the Cloud Run job."
  default     = ""
}