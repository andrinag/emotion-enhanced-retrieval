package com.example.myapplication

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.RadioButton
import androidx.recyclerview.widget.RecyclerView

class EvaluationAdapter(
    private val items: List<EvaluationDisplay>,
    private val onItemSelected: (EvaluationDisplay) -> Unit
) : RecyclerView.Adapter<EvaluationAdapter.ViewHolder>() {

    private var selectedPosition = -1

    inner class ViewHolder(val view: View) : RecyclerView.ViewHolder(view) {
        val radioButton: RadioButton = view.findViewById(R.id.radioButton)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.item_evaluation, parent, false)
        return ViewHolder(view)
    }

    override fun getItemCount(): Int = items.size

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val item = items[position]
        val text = buildString {
            append("${item.name} (${item.id})")
            item.taskName?.let { append("\nTask: $it") }
            item.taskType?.let { append(" [${item.taskType}]") }
        }
        holder.radioButton.text = text
        holder.radioButton.isChecked = position == selectedPosition

        holder.radioButton.setOnClickListener {
            selectedPosition = position
            notifyDataSetChanged()
            onItemSelected(item)
        }
    }
}
